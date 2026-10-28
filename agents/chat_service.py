"""대화형 채팅 서비스 (CLI / 웹 공통)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from agents.config import AppConfig, load_config
from agents.conversation_agent import ConversationalAgent
from agents.loader import load_drugs, load_symptoms
from agents.models import ConversationSession, Drug, Symptom


@dataclass
class ChatBundle:
    agent: ConversationalAgent
    session: ConversationSession


_drugs_cache: Optional[List[Drug]] = None
_symptoms_cache: Optional[List[Symptom]] = None


def _load_catalog(config: AppConfig) -> tuple[List[Drug], List[Symptom]]:
    global _drugs_cache, _symptoms_cache
    if _drugs_cache is None or _symptoms_cache is None:
        _drugs_cache = load_drugs(config.data_dir / "drugs.json")
        _symptoms_cache = load_symptoms(config.data_dir / "symptoms.json")
    return _drugs_cache, _symptoms_cache


from agents.trace_sink import create_trace_sink


def create_chat(config: Optional[AppConfig] = None) -> Tuple[ChatBundle, str]:
    """새 대화 세션을 시작하고 환영 메시지를 반환합니다."""
    cfg = config or load_config()
    drugs, symptoms = _load_catalog(cfg)
    sink = create_trace_sink(enabled=cfg.show_reasoning)
    agent = ConversationalAgent(drugs, symptoms, config=cfg, trace_sink=sink)
    opening = agent.start_session()
    return ChatBundle(agent=agent, session=opening.session), opening.message


def send_message(
    bundle: ChatBundle, user_message: str
) -> Tuple[ChatBundle, str, Optional[str], Optional[dict], Optional[dict]]:
    """사용자 메시지 1건 처리 → (bundle, reply, debug_trace, data_sources, reasoning_trace)."""
    response = bundle.agent.process_message(bundle.session, user_message)
    return (
        ChatBundle(agent=bundle.agent, session=response.session),
        response.message,
        response.debug_trace,
        response.data_sources,
        response.reasoning_trace,
    )


def is_reset_command(text: str) -> bool:
    norm = (text or "").strip().lower()
    return norm in {"reset", "새 대화", "처음부터", "new chat", "다시 시작"}
