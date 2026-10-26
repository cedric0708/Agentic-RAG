"""Multi-turn Conversational Agentic RAG 오케스트레이터."""

from __future__ import annotations

from typing import List, Optional

from agents.agent_trace import AgentStep, format_agent_trace, format_turn_reasoning_lines
from agents.config import AppConfig, load_config
from agents.conversation_formatter import (
    OFF_TOPIC_MESSAGE,
    WELCOME_MESSAGE,
    SLOT_QUESTIONS,
    format_early_advisory,
    format_first_symptom_ack,
    format_follow_up_advisory,
    format_follow_up_alternative,
    format_follow_up_safety,
    format_follow_up_why,
    format_follow_up_why_empty,
    format_mixed_input_note,
    format_recommendation_message,
    format_slot_ack,
    format_symptom_addon_ack,
    format_topic_change_ack,
    get_next_slot,
    is_ready_for_recommendation,
)
from agents.trace_sink import NullTraceSink, TraceSink
from agents.off_topic import has_off_topic_keyword, is_medical_related
from agents.slot_extractor import SlotExtractionResult
from agents.llm_client import extract_symptoms_llm, is_llm_enabled, polish_response_llm
from agents.llm_orchestrator import (
    TurnPlan,
    generate_rag_reply_llm,
    orchestrate_turn_llm,
)
from agents.models import UserInput
from agents.enhanced_pipeline import run_enhanced_pipeline_core
from agents.integrations.types import provenance_to_dict
from agents.intent_detector import UserIntent, detect_intent
from agents.models import (
    ConversationPhase,
    ConversationResponse,
    ConversationSession,
    ConversationTurn,
    Drug,
    PipelineResult,
    Symptom,
)
from agents.slot_extractor import (
    extract_slots_from_message,
    merge_user_input,
    pick_primary_symptoms,
    replace_primary_symptoms,
)
from agents.symptom_utils import dedupe_symptoms_for_chat
import os


def _use_llm_orchestrator() -> bool:
    if not is_llm_enabled():
        return False
    return os.getenv("USE_LLM_ORCHESTRATOR", "true").lower() in {"1", "true", "yes"}


class ConversationalAgent:
    """대화형 Agentic RAG 에이전트."""

    def __init__(
        self,
        drugs: List[Drug],
        symptoms: List[Symptom],
        config: Optional[AppConfig] = None,
        trace_sink: Optional[TraceSink] = None,
    ):
        self.drugs = drugs
        self.symptoms = symptoms
        self.symptoms_for_safety = dedupe_symptoms_for_chat(symptoms)
        self.config = config or load_config()
        self.trace_sink: TraceSink = trace_sink or NullTraceSink()
        self._last_pipeline_result: Optional[PipelineResult] = None

    def create_session(self) -> ConversationSession:
        return ConversationSession()

    def start_session(self) -> ConversationResponse:
        session = self.create_session()
        session.phase = ConversationPhase.COLLECTING
        session.turns.append(ConversationTurn(role="assistant", content=WELCOME_MESSAGE))
        return ConversationResponse(
            message=WELCOME_MESSAGE,
            session=session,
            phase=session.phase,
        )

    def _mark_slots_confirmed(self, session: ConversationSession, slot: Optional[str]) -> None:
        if not slot:
            return
        session.confirmed_slots.add(slot)
        session.pending_slot = None

    def _slot_was_answered(
        self,
        slot: str,
        extraction: SlotExtractionResult,
        message: str,
    ) -> bool:
        text = (message or "").strip()
        negative = extraction.is_negative_answer

        if slot == "symptoms":
            return bool(extraction.extracted_symptoms)
        if slot == "current_meds":
            return bool(extraction.extracted_drugs) or bool(
                extraction.user_input.current_drug_names
            ) or (negative and len(text.replace(" ", "")) <= 15)
        if slot == "allergies":
            return bool(extraction.extracted_allergies)
        if slot == "conditions":
            return bool(extraction.extracted_conditions)
        return False

    def _stream_turn_trace(self, intent: UserIntent, session: ConversationSession, notes: List[str]) -> None:
        if not self.config.show_reasoning:
            return
        self.trace_sink.begin()
        for line in format_turn_reasoning_lines(
            intent=intent.value,
            phase=session.phase.value,
            notes=notes,
        ):
            self.trace_sink.step(line)

    def _stream_pipeline_step(self, step: AgentStep) -> None:
        if not self.config.show_reasoning:
            return
        from agents.agent_trace import format_step_line

        self.trace_sink.begin()
        for line in format_step_line(step).splitlines():
            self.trace_sink.step(line)

    def _respond(
        self,
        session: ConversationSession,
        message: str,
        *,
        debug_trace: Optional[str] = None,
        data_sources: Optional[dict] = None,
    ) -> ConversationResponse:
        session.turns.append(ConversationTurn(role="assistant", content=message))
        trace = debug_trace
        sources = data_sources
        if self._last_pipeline_result:
            if self.config.enable_agent_trace:
                trace = format_agent_trace(self._last_pipeline_result.trace_steps)
            if sources is None and self._last_pipeline_result.data_provenance is not None:
                sources = provenance_to_dict(self._last_pipeline_result.data_provenance)
        self.trace_sink.end()
        return ConversationResponse(
            message=message,
            session=session,
            phase=session.phase,
            debug_trace=trace if self.config.enable_agent_trace else None,
            data_sources=sources,
        )

    def _apply_extraction(self, session: ConversationSession, message: str) -> SlotExtractionResult:
        pending = session.pending_slot
        extraction = extract_slots_from_message(
            message,
            self.drugs,
            self.symptoms,
            pending_slot=pending,
        )
        session.user_input = merge_user_input(session.user_input, extraction.user_input)

        if pending:
            if self._slot_was_answered(pending, extraction, message):
                self._mark_slots_confirmed(session, pending)
            elif extraction.extracted_symptoms:
                session.confirmed_slots.add("symptoms")
        elif extraction.extracted_symptoms:
            session.confirmed_slots.add("symptoms")
        elif extraction.extracted_drugs:
            session.confirmed_slots.add("current_meds")
        elif extraction.extracted_allergies:
            session.confirmed_slots.add("allergies")
        elif extraction.extracted_conditions:
            session.confirmed_slots.add("conditions")

        ui = session.user_input
        if ui.symptoms:
            session.confirmed_slots.add("symptoms")
        if ui.current_drug_ids or ui.current_drug_names:
            session.confirmed_slots.add("current_meds")
        if ui.allergies:
            session.confirmed_slots.add("allergies")
        if ui.conditions:
            session.confirmed_slots.add("conditions")

        return extraction

    def _ask_next_slot(self, session: ConversationSession) -> str:
        slot = get_next_slot(session)
        session.pending_slot = slot
        session.phase = ConversationPhase.CLARIFYING
        if slot:
            return SLOT_QUESTIONS[slot]
        return (
            "필요한 정보를 모두 확인했습니다. "
            "지금 일반의약품 후보를 검토해 드릴까요? (예: \"추천해 주세요\")"
        )

    def _latest_user_text(self, session: ConversationSession) -> str:
        for turn in reversed(session.turns):
            if turn.role == "user" and turn.content.strip():
                return turn.content.strip()
        return ""

    def _sync_confirmed_slots(self, session: ConversationSession) -> None:
        ui = session.user_input
        if ui.symptoms:
            session.confirmed_slots.add("symptoms")
        if ui.current_drug_ids or ui.current_drug_names:
            session.confirmed_slots.add("current_meds")
        if ui.allergies:
            session.confirmed_slots.add("allergies")
        if ui.conditions:
            session.confirmed_slots.add("conditions")

    def _apply_turn_plan(self, session: ConversationSession, plan: TurnPlan) -> None:
        patch = UserInput(symptoms=plan.symptoms or [])
        if plan.current_drug_names:
            patch.current_drug_names = plan.current_drug_names
        if plan.allergies:
            patch.allergies = plan.allergies
        if plan.conditions:
            patch.conditions = plan.conditions
        session.user_input = merge_user_input(session.user_input, patch)
        for slot in plan.slots_answered:
            session.confirmed_slots.add(slot)
        self._sync_confirmed_slots(session)
        session.pending_slot = get_next_slot(session)

    def _polish_message(self, draft: str, user_message: str, session: Optional[ConversationSession] = None, result: Optional[PipelineResult] = None) -> str:
        if result and session and _use_llm_orchestrator():
            refined = generate_rag_reply_llm(user_message, session, result, draft)
            if refined:
                return refined
        if is_llm_enabled():
            polished = polish_response_llm(draft, user_message)
            return polished or draft
        return draft

    def _process_message_llm(
        self,
        session: ConversationSession,
        text: str,
    ) -> Optional[ConversationResponse]:
        plan = orchestrate_turn_llm(
            text,
            session,
            has_recommendation=self._last_pipeline_result is not None
            and session.phase
            in {
                ConversationPhase.FOLLOW_UP,
                ConversationPhase.RECOMMENDING,
                ConversationPhase.EMERGENCY,
            },
        )
        if not plan:
            return None

        notes = [plan.reasoning] if plan.reasoning else ["LLM 턴 분석"]
        self._stream_turn_trace(UserIntent.PROVIDE_INFO, session, notes)

        if text.lower() in {"quit", "exit", "종료", "끝", "bye", "나가기"}:
            session.phase = ConversationPhase.ENDED
            return self._respond(session, "상담을 종료합니다. 증상이 지속되면 의료진 상담을 권장합니다.")

        if any(k in text for k in ("처음부터", "다시 시작", "리셋", "reset")):
            new_session = self.create_session()
            new_session.phase = ConversationPhase.COLLECTING
            self._last_pipeline_result = None
            msg = WELCOME_MESSAGE + "\n\n(대화를 새로 시작합니다.)"
            new_session.turns.append(ConversationTurn(role="assistant", content=msg))
            return ConversationResponse(message=msg, session=new_session, phase=new_session.phase)

        if plan.is_off_topic and not (plan.symptoms or plan.current_drug_names or plan.conditions):
            msg = plan.reply or OFF_TOPIC_MESSAGE
            return self._respond(session, msg)

        self._apply_turn_plan(session, plan)

        if plan.is_emergency:
            return self._run_recommendation(session, user_text=text)

        if self._last_pipeline_result and not plan.run_recommendation:
            if plan.reply:
                session.phase = ConversationPhase.FOLLOW_UP
                return self._respond(session, plan.reply)

        if plan.run_recommendation and session.user_input.symptoms:
            return self._run_recommendation(session, user_text=text)

        if is_ready_for_recommendation(session):
            return self._run_recommendation(session)

        msg = plan.reply
        if not msg:
            next_slot = get_next_slot(session)
            msg = SLOT_QUESTIONS.get(next_slot or "symptoms", WELCOME_MESSAGE)
        session.phase = ConversationPhase.CLARIFYING
        return self._respond(session, msg)

    def _resolve_symptoms_from_message(self, message: str) -> List[str]:
        extraction = extract_slots_from_message(
            message,
            self.drugs,
            self.symptoms,
            pending_slot=None,
        )
        if extraction.extracted_symptoms:
            return extraction.extracted_symptoms

        if is_llm_enabled():
            known = sorted({s.name for s in self.symptoms if s.name})
            llm_symptoms = extract_symptoms_llm(message, known)
            if llm_symptoms:
                return pick_primary_symptoms(llm_symptoms, message)
        return []

    def _run_recommendation(
        self,
        session: ConversationSession,
        *,
        topic_ack: str = "",
        user_text: str = "",
    ) -> ConversationResponse:
        session.phase = ConversationPhase.RECOMMENDING
        context = user_text or self._latest_user_text(session)
        if self.config.show_reasoning:
            self.trace_sink.begin()
        result = run_enhanced_pipeline_core(
            session.user_input,
            self.drugs,
            self.symptoms_for_safety,
            top_k=self.config.top_k,
            config=self.config,
            context_text=context,
            on_step=self._stream_pipeline_step if self.config.show_reasoning else None,
        )
        self._last_pipeline_result = result

        if result.is_emergency:
            session.phase = ConversationPhase.EMERGENCY
            message = result.emergency_message
        else:
            session.phase = ConversationPhase.FOLLOW_UP
            draft = format_recommendation_message(result, topic_ack=topic_ack)
            message = self._polish_message(
                draft,
                user_text or context,
                session=session,
                result=result,
            )
            session.last_pipeline_summary = message

        debug_trace = (
            format_agent_trace(result.trace_steps)
            if self.config.enable_agent_trace
            else None
        )
        data_sources = provenance_to_dict(result.data_provenance)
        self.trace_sink.end()
        session.turns.append(ConversationTurn(role="assistant", content=message))
        return ConversationResponse(
            message=message,
            session=session,
            phase=session.phase,
            debug_trace=debug_trace,
            data_sources=data_sources,
        )

    def _handle_follow_up(self, session: ConversationSession, intent: UserIntent) -> str:
        result = self._last_pipeline_result
        if result is None:
            return "아직 추천 결과가 없습니다. 증상을 먼저 알려주시겠어요?"

        if intent == UserIntent.FOLLOW_UP_WHY:
            if not result.decision.recommended:
                return format_follow_up_why_empty(result, self._latest_user_text(session))
            return format_follow_up_why(result, self._latest_user_text(session))
        if intent == UserIntent.FOLLOW_UP_ALTERNATIVE:
            return format_follow_up_alternative(result)
        if intent == UserIntent.FOLLOW_UP_SAFETY:
            return format_follow_up_safety(result)
        if intent == UserIntent.FOLLOW_UP_ADVISORY:
            return format_follow_up_advisory(result, self._latest_user_text(session))
        return (
            "추천 결과에 대해 \"왜 이 약인가요?\", \"다른 약은요?\", "
            "\"부작용이 뭐예요?\"처럼 질문해 주세요."
        )

    def process_message(
        self,
        session: ConversationSession,
        user_message: str,
    ) -> ConversationResponse:
        """사용자 메시지 1턴을 처리합니다."""
        text = (user_message or "").strip()
        session.turn_count += 1
        session.turns.append(ConversationTurn(role="user", content=text))

        if not text:
            message = "메시지를 입력해 주세요."
            session.turns.append(ConversationTurn(role="assistant", content=message))
            return ConversationResponse(message=message, session=session, phase=session.phase)

        if _use_llm_orchestrator():
            llm_response = self._process_message_llm(session, text)
            if llm_response is not None:
                return llm_response

        has_prior_result = self._last_pipeline_result is not None
        intent = detect_intent(
            text,
            has_recommendation=has_prior_result
            and session.phase
            in {
                ConversationPhase.FOLLOW_UP,
                ConversationPhase.RECOMMENDING,
                ConversationPhase.EMERGENCY,
            },
            pending_slot=session.pending_slot,
        )

        if intent == UserIntent.OFF_TOPIC:
            if has_off_topic_keyword(text) and is_medical_related(text):
                mixed = format_mixed_input_note(text)
                new_symptoms = self._resolve_symptoms_from_message(text)
                if new_symptoms:
                    session.user_input = replace_primary_symptoms(
                        session.user_input,
                        new_symptoms,
                        user_text=text,
                    )
                    session.confirmed_slots.add("symptoms")
                    return self._run_recommendation(
                        session,
                        topic_ack=mixed,
                        user_text=text,
                    )
            message = OFF_TOPIC_MESSAGE
            session.turns.append(ConversationTurn(role="assistant", content=message))
            return ConversationResponse(message=message, session=session, phase=session.phase)

        if intent == UserIntent.FOLLOW_UP_ADVISORY and not has_prior_result:
            self._apply_extraction(session, text)
            advisory = format_early_advisory(text)
            if advisory:
                pending = session.pending_slot or get_next_slot(session)
                if pending:
                    session.pending_slot = pending
                    message = advisory + "\n\n" + SLOT_QUESTIONS[pending]
                else:
                    message = advisory
                session.phase = ConversationPhase.CLARIFYING
                session.turns.append(ConversationTurn(role="assistant", content=message))
                return ConversationResponse(message=message, session=session, phase=session.phase)

        if intent == UserIntent.EXIT:
            session.phase = ConversationPhase.ENDED
            message = "상담을 종료합니다. 증상이 지속되면 의료진 상담을 권장합니다."
            session.turns.append(ConversationTurn(role="assistant", content=message))
            return ConversationResponse(message=message, session=session, phase=session.phase)

        if intent == UserIntent.RESET:
            new_session = self.create_session()
            new_session.phase = ConversationPhase.COLLECTING
            self._last_pipeline_result = None
            message = WELCOME_MESSAGE + "\n\n(대화를 새로 시작합니다.)"
            new_session.turns.append(ConversationTurn(role="assistant", content=message))
            return ConversationResponse(message=message, session=new_session, phase=new_session.phase)

        if session.phase in {
            ConversationPhase.FOLLOW_UP,
            ConversationPhase.RECOMMENDING,
            ConversationPhase.EMERGENCY,
        }:
            if intent in {
                UserIntent.FOLLOW_UP_WHY,
                UserIntent.FOLLOW_UP_ALTERNATIVE,
                UserIntent.FOLLOW_UP_SAFETY,
                UserIntent.FOLLOW_UP_ADVISORY,
            }:
                message = self._handle_follow_up(session, intent)
                self._stream_turn_trace(intent, session, [f"follow-up ({intent.value})"])
                return self._respond(session, message)

            if intent in {UserIntent.SYMPTOM_CHANGE, UserIntent.REQUEST_RECOMMENDATION}:
                old_symptoms = list(session.user_input.symptoms)
                new_symptoms = self._resolve_symptoms_from_message(text)
                if new_symptoms:
                    session.user_input = replace_primary_symptoms(
                        session.user_input,
                        new_symptoms,
                        user_text=text,
                    )
                    session.confirmed_slots.add("symptoms")
                else:
                    self._apply_extraction(session, text)

                topic_ack = format_topic_change_ack(old_symptoms, session.user_input.symptoms)
                topic_ack = format_mixed_input_note(text) + topic_ack
                if session.user_input.symptoms:
                    return self._run_recommendation(
                        session,
                        topic_ack=topic_ack,
                        user_text=text,
                    )

        if session.phase == ConversationPhase.GREETING:
            session.phase = ConversationPhase.COLLECTING

        prior_symptoms = list(session.user_input.symptoms)
        confirmed_before = set(session.confirmed_slots)
        extraction = self._apply_extraction(session, text)
        newly_confirmed = session.confirmed_slots - confirmed_before

        if intent == UserIntent.REQUEST_RECOMMENDATION or is_ready_for_recommendation(session):
            if not session.user_input.symptoms:
                message = SLOT_QUESTIONS["symptoms"]
                session.pending_slot = "symptoms"
                session.phase = ConversationPhase.CLARIFYING
            else:
                for slot in ("symptoms", "current_meds", "allergies", "conditions"):
                    if slot in session.confirmed_slots:
                        continue
                    if slot == "symptoms" and session.user_input.symptoms:
                        session.confirmed_slots.add(slot)
                    elif slot == "current_meds" and (
                        session.user_input.current_drug_ids
                        or session.user_input.current_drug_names
                    ):
                        session.confirmed_slots.add(slot)
                    elif slot == "allergies" and session.user_input.allergies:
                        session.confirmed_slots.add(slot)
                    elif slot == "conditions" and session.user_input.conditions:
                        session.confirmed_slots.add(slot)

                if is_ready_for_recommendation(session):
                    return self._run_recommendation(session)

                message = self._ask_next_slot(session)
                self._stream_turn_trace(intent, session, extraction.messages[:2])
                return self._respond(session, message)

        next_slot = get_next_slot(session)
        if next_slot:
            ack_parts: List[str] = []
            if newly_confirmed:
                for slot in ("conditions", "allergies", "current_meds", "symptoms"):
                    if slot in newly_confirmed:
                        slot_ack = format_slot_ack(session, slot)
                        if slot_ack:
                            ack_parts.append(slot_ack)
                        break
            elif extraction.extracted_symptoms:
                if prior_symptoms:
                    addon = format_symptom_addon_ack(extraction.extracted_symptoms)
                else:
                    addon = format_first_symptom_ack(session.user_input.symptoms)
                if addon:
                    ack_parts.append(addon)
            ack = "\n\n".join(ack_parts)
            message = (ack + "\n\n" if ack else "") + SLOT_QUESTIONS[next_slot]
            session.pending_slot = next_slot
            session.phase = ConversationPhase.CLARIFYING
        else:
            session.phase = ConversationPhase.CLARIFYING
            message = (
                "정보 확인이 거의 끝났습니다. "
                "일반의약품 후보를 검토해 드릴까요? (\"추천해 주세요\"라고 말씀해 주세요)"
            )

        notes: List[str] = []
        if extraction.extracted_symptoms:
            notes.append(f"증상 반영: {', '.join(extraction.extracted_symptoms)}")
        if extraction.extracted_conditions:
            notes.append(f"기저상태 반영: {', '.join(extraction.extracted_conditions)}")
        if extraction.messages:
            notes.extend(extraction.messages[:2])
        self._stream_turn_trace(intent, session, notes)
        return self._respond(session, message)


def run_conversational_turn(
    session: ConversationSession,
    user_message: str,
    drugs: List[Drug],
    symptoms: List[Symptom],
    config: Optional[AppConfig] = None,
    agent: Optional[ConversationalAgent] = None,
) -> ConversationResponse:
    """대화 1턴 실행 API."""
    bot = agent or ConversationalAgent(drugs, symptoms, config=config)
    return bot.process_message(session, user_message)
