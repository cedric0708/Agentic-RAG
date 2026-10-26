# api/diagnosis.py
from fastapi import APIRouter, Request, Response, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid

# AI 담당자가 제공한 대화형 서비스 모듈 및 영속성 레이어
from agents.chat_service import create_chat, send_message, ChatBundle
from agents.persistence import save_session_to_file, load_session_from_file

router = APIRouter(prefix="/api")

class ChatRequest(BaseModel):
    content: str
    chat_id: str  # 프론트엔드에서 관리하는 채팅방 식별자

# 서버 가용성 향상을 위한 메모리 내 세션 캐시
SESSION_DB: Dict[str, ChatBundle] = {}

@router.post("/diagnosis")
async def predict_symptom(request: Request, response: Response, body: ChatRequest):
    user_msg = body.content
    chat_id = body.chat_id

    # 1. 쿠키에서 유저 식별자(user_id) 추출
    user_id = request.cookies.get("user_id")

    # 신규 유저인 경우 고유 ID 생성 및 쿠키 발급
    if not user_id:
        user_id = str(uuid.uuid4())
        # 로컬 HTTP 및 동일 출처 Vite proxy에서 사용하는 세션 쿠키
        response.set_cookie(
            key="user_id", 
            value=user_id, 
            max_age=3600*24*30, # 30일 유지
            samesite="lax",
            secure=False,
            httponly=True
        )
    
    # 유저 ID와 채팅방 ID를 조합하여 고유 세션 키 생성
    session_key = f"{user_id}_{chat_id}"

    try:
        # 2. 대화 세션 로드 (메모리 우선 -> 파일 시스템 순)
        if session_key not in SESSION_DB:
            saved_session = load_session_from_file(session_key)

            if saved_session:
                # 저장된 세션이 있다면 복원 (Agent 인스턴스는 새로 생성)
                bundle, _ = create_chat()
                bundle.session = saved_session
                SESSION_DB[session_key] = bundle
            else:
                # 신규 대화인 경우 최초 1회 초기화
                bundle, _ = create_chat()
                SESSION_DB[session_key] = bundle

        current_bundle = SESSION_DB[session_key]

        # 3. AI 에이전트를 통한 메시지 처리
        # (OpenAI gpt-4o-mini 모델을 사용하여 맥락 분석 및 답변 생성)
        updated_bundle, assistant_response, debug_trace, data_sources = send_message(
            current_bundle, user_msg
        )
        
        # 4. 상태 갱신 및 영속성 레이어에 저장
        SESSION_DB[session_key] = updated_bundle
        save_session_to_file(session_key, updated_bundle.session)
        
        # 5. 최종 응답 반환
        payload = {
            "role": "ai",
            "content": assistant_response,
            "debug_trace": debug_trace,
        }
        if data_sources is not None:
            payload["data_sources"] = data_sources
        return payload

    except Exception as e:
        # 시스템 에러 발생 시 상세 로깅
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI Assistant Error: {str(e)}")
