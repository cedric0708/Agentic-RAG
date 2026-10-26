from __future__ import annotations

from typing import Callable, List, Optional

from agents.agent_trace import AgentStep, format_agent_trace
from agents.clarification_agent import generate_clarifying_questions
from agents.config import AppConfig, load_config
from agents.generator import build_emergency_response
from agents.models import ChromaEvidence, Drug, PipelineResult, Symptom, UserInput
from agents.query_agent import analyze_user_input
from agents.red_flag import detect_red_flags
from agents.recommendation_agent import decide_recommendations
from agents.reranker import rerank_results
from agents.integrations.context import build_de_context
from agents.integrations.types import DataBackend, DataProvenance
from agents.retriever import retrieve_candidates, retrieve_symptom_context
from agents.safety import check_symptom_red_flags, evaluate_drug_safety
from agents.validator import validate_decision


def _format_enhanced_response(
    user_input: UserInput,
    decision,
    validation,
    trace_steps: List[AgentStep],
    use_chroma: bool = True,
) -> str:
    lines: List[str] = []
    lines.append(format_agent_trace(trace_steps))
    lines.append("\n[입력 정보]")
    lines.append(f"- 증상: {', '.join(user_input.symptoms) if user_input.symptoms else '없음'}")
    lines.append(f"- 현재 복용 약 ID: {', '.join(user_input.current_drug_ids) if user_input.current_drug_ids else '없음'}")
    lines.append(f"- 현재 복용 약 이름: {', '.join(user_input.current_drug_names) if user_input.current_drug_names else '없음'}")
    lines.append(f"- 알레르기: {', '.join(user_input.allergies) if user_input.allergies else '없음'}")
    lines.append(f"- 기저/특이 상태: {', '.join(user_input.conditions) if user_input.conditions else '없음'}")

    lines.append("\n[최종 추천 후보]")
    if not decision.recommended:
        lines.append("- 안전성 기준을 통과한 추천 후보가 없습니다. 약사/의료진 상담을 권장합니다.")
    for idx, item in enumerate(decision.recommended, start=1):
        result = item.retrieval
        drug = result.drug
        lines.append(f"{idx}. {drug.name_ko} ({drug.drug_id})")
        lines.append(f"   - 최종 점수: {item.final_score:.1f}")
        lines.append(f"   - 신뢰도: {item.confidence * 100:.0f}%")
        lines.append(f"   - 위험도: {item.risk_level}")
        lines.append(f"   - 적응증: {drug.indications}")
        lines.append(f"   - 복용법: {drug.dosage}")
        lines.append(f"   - 매칭 증상: {', '.join(result.matched_symptoms) if result.matched_symptoms else '없음'}")
        lines.append(f"   - 근거: {'; '.join(item.rerank_reasons)}")
        if result.chroma_evidence:
            top_hit = max(result.chroma_evidence, key=lambda x: x.relevance)
            lines.append(
                f"   - ChromaDB: {top_hit.chunk_id} "
                f"(유사도 {top_hit.relevance:.2f})"
            )

    lines.append("\n[반대 추천 / 제외 후보]")
    if not decision.rejected:
        lines.append("- 제외된 후보 없음")
    for item in decision.rejected:
        drug = item.retrieval.drug
        lines.append(f"- {drug.name_ko} ({drug.drug_id})")
        lines.append(f"  - 제외/후순위 이유: 위험도={item.risk_level}, 최종점수={item.final_score:.1f}")
        if item.safety_penalty:
            lines.append(f"  - 안전성 penalty: -{item.safety_penalty:.1f}")

    lines.append("\n[Validator]")
    lines.append("- 검증 결과: " + ("통과" if validation.passed else "실패"))
    for check in validation.checks:
        lines.append(f"  ✔ {check}")
    for error in validation.errors:
        lines.append(f"  ✘ {error}")

    lines.append("\n[주의]")
    lines.append("- Retrieval/Safety 출처는 파이프라인 data_provenance 및 API data_sources 필드를 참고하세요.")
    if use_chroma:
        lines.append("- (legacy) ChromaDB hybrid 사용 가능.")
    lines.append("- 실제 복약 판단은 의사/약사 상담이 우선입니다.")
    return "\n".join(lines)


def run_enhanced_pipeline_core(
    user_input: UserInput,
    drugs: List[Drug],
    symptoms: List[Symptom],
    top_k: int = 3,
    config: Optional[AppConfig] = None,
    *,
    context_text: str = "",
    on_step: Optional[Callable[[AgentStep], None]] = None,
) -> PipelineResult:
    """Agentic RAG 파이프라인을 실행하고 구조화된 결과를 반환합니다."""
    cfg = config or load_config()
    top_k = top_k or cfg.top_k
    trace_steps: List[AgentStep] = []
    symptom_context: List[ChromaEvidence] = []
    provenance = DataProvenance()
    de_ctx = build_de_context(cfg)

    def emit(step: AgentStep) -> None:
        trace_steps.append(step)
        if on_step:
            on_step(step)

    query_analysis = analyze_user_input(user_input, drugs, symptoms)
    emit(AgentStep("Query Agent", "입력 분석 완료", query_analysis.messages))

    clarification = generate_clarifying_questions(user_input)
    if clarification.needs_clarification:
        emit(
            AgentStep(
                "Clarification Agent",
                "추가 확인 필요",
                [f"Q. {q}" for q in clarification.questions[:3]]
                + [f"Reason: {r}" for r in clarification.reasons[:2]],
            )
        )
    else:
        emit(
            AgentStep(
                "Clarification Agent",
                "추가 질문 불필요",
                ["추천에 필요한 기본 정보가 입력되었습니다."],
            )
        )

    red_flag = detect_red_flags(user_input, symptoms, user_text=context_text)
    emit(AgentStep("Safety Agent", "red flag 검사 완료", red_flag.reasons or [red_flag.action]))
    if red_flag.has_red_flag:
        global_safety = check_symptom_red_flags(
            user_input, symptoms, user_text=context_text
        )
        emergency = build_emergency_response(user_input, global_safety)
        de_ctx.close()
        return PipelineResult(
            user_input=user_input,
            trace_steps=trace_steps,
            decision=decide_recommendations([]),
            validation=validate_decision(decide_recommendations([]), drugs),
            is_emergency=True,
            emergency_message=emergency,
            data_provenance=provenance,
        )

    if cfg.use_chroma:
        symptom_context = retrieve_symptom_context(
            user_input,
            top_n=2,
            user_text=context_text,
        )
        if symptom_context:
            emit(
                AgentStep(
                    "ChromaDB",
                    "증상 참고 청크 검색",
                    [f"{item.chunk_id}: {item.document[:80]}..." for item in symptom_context],
                )
            )

    try:
        retrieved, retrieval_backend, retrieval_notes = retrieve_candidates(
            user_input,
            drugs,
            top_k=top_k,
            config=cfg,
            de_ctx=de_ctx,
            user_text=context_text,
            use_chroma=cfg.use_chroma,
            chroma_top_n=cfg.chroma_top_n,
            chroma_weight=cfg.chroma_score_weight,
            min_score=cfg.min_retrieval_score,
        )
        provenance.retrieval = retrieval_backend
        for note in retrieval_notes:
            provenance.note(note)
    finally:
        pass

    retrieval_label = provenance.retrieval.value
    emit(
        AgentStep(
            "Retrieval Agent",
            f"후보 {len(retrieved)}개 검색 완료 [source={retrieval_label}]",
            retrieval_notes
            + [
                f"{item.drug.name_ko}: score={item.score:.1f}, src={item.retrieval_source}"
                + (f", de={item.de_score:.2f}" if item.de_score is not None else "")
                + (f", chroma={len(item.chroma_evidence)}" if item.chroma_evidence else "")
                for item in retrieved
            ]
            or ["관련 후보 없음"],
        )
    )

    _ = [evaluate_drug_safety(user_input, result.drug) for result in retrieved]

    reranked = rerank_results(
        user_input,
        retrieved,
        drugs,
        graph_client=de_ctx.graph,
        de_fallback_legacy=de_ctx.fallback_legacy,
    )
    for item in reranked:
        for reason in item.rerank_reasons:
            if reason.startswith("safety_backend="):
                raw = reason.split("=", 1)[1].strip()
                try:
                    provenance.safety = DataBackend(raw)
                except ValueError:
                    provenance.note(f"unknown safety_backend={raw}")
    emit(
        AgentStep(
            "Reranker",
            f"관련도 + 안전성 재정렬 [safety_source={provenance.safety.value}]",
            [
                f"{item.retrieval.drug.name_ko}: final={item.final_score:.1f}, risk={item.risk_level}, blocked={item.blocked}"
                for item in reranked
            ]
            or ["재정렬할 후보 없음"],
        )
    )

    decision = decide_recommendations(reranked, max_recommendations=top_k)
    emit(AgentStep("Recommendation Agent", "추천/비추천 판단 완료", decision.messages))

    validation = validate_decision(decision, drugs)
    emit(
        AgentStep(
            "Validator",
            "출력 검증 완료" if validation.passed else "출력 검증 실패",
            validation.checks[:4] + validation.errors[:4],
        )
    )

    provenance.note(
        f"summary retrieval={provenance.retrieval.value} safety={provenance.safety.value}"
    )
    de_ctx.close()
    return PipelineResult(
        user_input=user_input,
        trace_steps=trace_steps,
        decision=decision,
        validation=validation,
        symptom_context=symptom_context,
        data_provenance=provenance,
    )


def run_enhanced_pipeline(
    user_input: UserInput,
    drugs: List[Drug],
    symptoms: List[Symptom],
    top_k: int = 3,
    config: Optional[AppConfig] = None,
) -> str:
    """기존 단답형 CLI용 문자열 응답 (하위 호환)."""
    cfg = config or load_config()
    result = run_enhanced_pipeline_core(user_input, drugs, symptoms, top_k=top_k, config=cfg)

    if result.is_emergency:
        return format_agent_trace(result.trace_steps) + "\n\n" + result.emergency_message

    return _format_enhanced_response(
        result.user_input,
        result.decision,
        result.validation,
        result.trace_steps,
        use_chroma=cfg.use_chroma,
    )
