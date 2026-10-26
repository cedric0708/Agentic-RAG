import json
import os
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Any, Dict
from agents.models import ConversationSession, UserInput, ConversationTurn, ConversationPhase

SESSION_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"

def session_to_dict(session: ConversationSession) -> Dict[str, Any]:
    """ConversationSession 객체를 JSON 저장이 가능한 딕셔너리로 변환."""
    d = asdict(session)
    # set 객체는 JSON 직렬화가 안 되므로 list로 변환
    d["confirmed_slots"] = list(session.confirmed_slots)
    # Enum 값 처리
    d["phase"] = session.phase.value
    return d

def dict_to_session(d: Dict[str, Any]) -> ConversationSession:
    """딕셔너리 데이터를 ConversationSession 객체로 복원."""
    user_input_data = d.get("user_input", {})
    user_input = UserInput(**user_input_data)
    
    turns_data = d.get("turns", [])
    turns = [ConversationTurn(**t) for t in turns_data]
    
    confirmed_slots = set(d.get("confirmed_slots", []))
    phase = ConversationPhase(d.get("phase", ConversationPhase.GREETING.value))
    
    return ConversationSession(
        user_input=user_input,
        phase=phase,
        turns=turns,
        confirmed_slots=confirmed_slots,
        pending_slot=d.get("pending_slot"),
        turn_count=d.get("turn_count", 0),
        last_pipeline_summary=d.get("last_pipeline_summary")
    )

def save_session_to_file(user_id: str, session: ConversationSession):
    """유저 세션을 JSON 파일로 저장."""
    os.makedirs(SESSION_DIR, exist_ok=True)
    
    file_path = os.path.join(SESSION_DIR, f"{user_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(session_to_dict(session), f, ensure_ascii=False, indent=2)

def load_session_from_file(user_id: str) -> ConversationSession | None:
    """JSON 파일에서 유저 세션 복원."""
    file_path = os.path.join(SESSION_DIR, f"{user_id}.json")
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return dict_to_session(data)
    except Exception as e:
        print(f"Error loading session for {user_id}: {e}")
        return None
