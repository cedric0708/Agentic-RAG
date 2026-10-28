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
from agents.integrations.types import DataBackend, DataProvenance, provenance_to_dict
from agents.reasoning_trace import (
    PipelineTraceBridge,
    apply_safety_provenance,
    compute_retrieval_data_sources,
    legacy_messages_clarification,
    legacy_messages_query,
    recommendation_decision_trace,
    reranker_evidence_from_results,
    retrieval_evidence_from_results,
    user_input_summary,
)
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
    bridge = PipelineTraceBridge.create(cfg, trace_steps, on_step)

    def _finish(
        *,
        pipeline_status: str,
        decision,
        validation,
        is_emergency: bool = False,
        emergency_message: str = "",
    ) -> PipelineResult:
        provenance.note(
            f"summary retrieval={provenance.retrieval.value} safety={provenance.safety.value}"
        )
        reasoning_dict = bridge.finalize(
            pipeline_status=pipeline_status,
            data_provenance=provenance_to_dict(provenance) or {},
        )
        de_ctx.close()
        return PipelineResult(
            user_input=user_input,
            trace_steps=trace_steps,
            decision=decision,
            validation=validation,
            is_emergency=is_emergency,
            emergency_message=emergency_message,
            symptom_context=symptom_context,
            data_provenance=provenance,
            reasoning_trace=reasoning_dict,
        )

    # --- Query Agent ---
    query_analysis = analyze_user_input(user_input, drugs, symptoms)
    q_status, q_messages = legacy_messages_query(query_analysis)

    def _populate_query(t) -> None:
        t.input_summary = user_input_summary(user_input)
        if context_text.strip():
            t.input_summary["context_text"] = context_text.strip()[:200]
        t.decision = {
            "detected_terms_count": len(query_analysis.detected_terms),
            "unknown_terms": query_analysis.unknown_terms[:8],
        }
        t.data_sources = ["legacy_json"]
        t.summary = "사용자 입력 및 증상·조건 분석 완료"
        if query_analysis.unknown_terms:
            t.warnings.append("일부 입력 용어가 catalog에서 확인되지 않음")

    bridge.emit_step(
        step_id="query_agent",
        agent="Query Agent",
        legacy_status=q_status,
        legacy_messages=q_messages,
        populate=_populate_query if bridge.enable_structured else None,
    )

    # --- Clarification Agent ---
    clarification = generate_clarifying_questions(user_input)
    c_status, c_messages = legacy_messages_clarification(clarification)

    def _populate_clarification(t) -> None:
        t.input_summary = user_input_summary(user_input)
        t.decision = {
            "needs_clarification": clarification.needs_clarification,
            "questions": clarification.questions[:5],
            "reasons": clarification.reasons[:5],
        }
        t.summary = c_status

    bridge.emit_step(
        step_id="clarification_agent",
        agent="Clarification Agent",
        legacy_status=c_status,
        legacy_messages=c_messages,
        populate=_populate_clarification if bridge.enable_structured else None,
    )

    # --- Safety Agent ---
    red_flag = detect_red_flags(user_input, symptoms, user_text=context_text)
    safe_messages = [
        f"red_flag={red_flag.has_red_flag}",
        f"action={red_flag.action}",
    ]
    if red_flag.matched:
        safe_messages.append(f"signals={', '.join(red_flag.matched[:5])}")

    def _populate_safety(t) -> None:
        t.input_summary = user_input_summary(user_input)
        t.decision = {
            "has_red_flag": red_flag.has_red_flag,
            "matched_signals": red_flag.matched,
            "action": red_flag.action,
        }
        t.data_sources = ["legacy_json"]
        t.summary = "응급 red flag 검사 완료"
        if red_flag.has_red_flag:
            t.status = "warning"
            t.warnings.append("응급 징후 가능 — 추천 파이프라인 조기 종료")

    bridge.emit_step(
        step_id="safety_agent",
        agent="Safety Agent",
        legacy_status="응급 red flag 검사 완료",
        legacy_messages=safe_messages,
        populate=_populate_safety if bridge.enable_structured else None,
    )

    if red_flag.has_red_flag:
        global_safety = check_symptom_red_flags(
            user_input, symptoms, user_text=context_text
        )
        emergency = build_emergency_response(user_input, global_safety)
        bridge.add_skipped_pipeline_tail("retrieval_agent")
        empty_decision = decide_recommendations([])
        return _finish(
            pipeline_status="emergency_stop",
            decision=empty_decision,
            validation=validate_decision(empty_decision, drugs),
            is_emergency=True,
            emergency_message=emergency,
        )

    if cfg.use_chroma:
        symptom_context = retrieve_symptom_context(
            user_input,
            top_n=2,
            user_text=context_text,
        )
        chroma_messages: List[str] = []
        if symptom_context:
            chroma_messages.append(f"Chroma hits={len(symptom_context)}")
        else:
            chroma_messages.append("ChromaDB 참고 청크 없음")

        def _populate_chroma(t) -> None:
            t.data_sources = ["chroma"]
            if symptom_context:
                t.decision = {"hits": len(symptom_context)}
                t.evidence = [
                    {"chunk_id": item.chunk_id, "relevance": round(item.relevance, 3)}
                    for item in symptom_context[:3]
                ]
                t.summary = "증상 참고 청크 검색 완료"
            else:
                t.status = "skipped"
                t.summary = "ChromaDB 참고 청크 없음"

        bridge.emit_step(
            step_id="chromadb",
            agent="ChromaDB",
            legacy_status="증상 참고 청크 검색",
            legacy_messages=chroma_messages,
            populate=_populate_chroma if bridge.enable_structured else None,
        )

    # --- Retrieval Agent ---
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
    hit_sources = compute_retrieval_data_sources(retrieved, retrieval_backend, retrieval_notes)
    provenance.retrieval_hit_sources = list(hit_sources)
    for note in retrieval_notes:
        provenance.note(note)

    ret_messages = [
        f"후보 {len(retrieved)}개 검색 [run_backend={retrieval_backend.value}]",
        f"hit_sources={','.join(hit_sources) if hit_sources else 'none'}",
    ]
    stub_count = sum(1 for r in retrieved if r.catalog_match == "stub")
    if stub_count:
        ret_messages.append(f"stub catalog 미매칭 후보 {stub_count}건")

    def _populate_retrieval(t) -> None:
        t.input_summary = user_input_summary(user_input)
        t.decision = {
            "retrieved_count": len(retrieved),
            "backend": retrieval_backend.value,
            "hit_sources": hit_sources,
            "fallback_notes": retrieval_notes[:6],
        }
        t.evidence = retrieval_evidence_from_results(retrieved)
        t.data_sources = list(hit_sources)
        t.summary = f"후보 {len(retrieved)}개 검색 [source={retrieval_backend.value}]"
        if stub_count:
            t.warnings.append(
                f"stub catalog 미매칭 후보 {stub_count}건 — 추천 전 추가 검증 필요"
            )
        if any("fallback" in n.lower() or "실패" in n for n in retrieval_notes):
            t.warnings.append("PostgreSQL 검색 실패 또는 fallback 발생 — legacy 출처 포함")

    bridge.emit_step(
        step_id="retrieval_agent",
        agent="Retrieval Agent",
        legacy_status="검색 완료",
        legacy_messages=ret_messages,
        populate=_populate_retrieval if bridge.enable_structured else None,
    )

    _ = [evaluate_drug_safety(user_input, result.drug) for result in retrieved]

    # --- Reranker ---
    reranked = rerank_results(
        user_input,
        retrieved,
        drugs,
        graph_client=de_ctx.graph,
        de_fallback_legacy=de_ctx.fallback_legacy,
    )
    apply_safety_provenance(provenance, reranked)
    safety_sources = list(provenance.safety_backends) or [provenance.safety.value]
    rerank_messages = [
        f"candidate_count={len(reranked)}",
        f"safety_backends={','.join(safety_sources)}",
    ]

    def _populate_rerank(t) -> None:
        t.decision = {
            "candidate_count": len(reranked),
            "safety_backend": provenance.safety.value,
            "safety_backends": list(provenance.safety_backends),
        }
        t.evidence = reranker_evidence_from_results(reranked)
        t.data_sources = list(dict.fromkeys(safety_sources + hit_sources))
        t.summary = f"관련성·안전성 재정렬 (safety={provenance.safety.value})"
        for ev in t.evidence:
            v = ev.get("neo4j_verdict")
            if v in ("UNKNOWN", "NOT_FOUND"):
                t.warnings.append(
                    f"{ev.get('drug_name')}: Neo4j {v} — 금기 확정 아님, 미확인/미발견"
                )
            if ev.get("needs_extra_verification"):
                t.warnings.append(
                    f"{ev.get('drug_name')}: 추가 검증 필요(stub 또는 Neo4j 미검증)"
                )

    bridge.emit_step(
        step_id="reranker",
        agent="Reranker",
        legacy_status="재정렬 완료",
        legacy_messages=rerank_messages,
        populate=_populate_rerank if bridge.enable_structured else None,
    )

    # --- Recommendation Agent ---
    decision = decide_recommendations(reranked, max_recommendations=top_k)
    rec_messages = list(decision.messages) if decision.messages else []
    rec_messages.insert(
        0,
        f"추천 {len(decision.recommended)}건 / 제외 {len(decision.rejected)}건",
    )

    def _populate_rec(t) -> None:
        t.decision = recommendation_decision_trace(decision)
        t.summary = (
            f"추천 {len(decision.recommended)}건 / 제외 {len(decision.rejected)}건"
        )
        if not decision.recommended:
            t.warnings.append("확정 추천 후보 없음 — stub·안전 미검증·차단 등 사유 가능")

    bridge.emit_step(
        step_id="recommendation_agent",
        agent="Recommendation Agent",
        legacy_status="추천 결정",
        legacy_messages=rec_messages,
        populate=_populate_rec if bridge.enable_structured else None,
    )

    # --- Validator ---
    validation = validate_decision(decision, drugs)
    val_messages = [
        "출력 검증 완료" if validation.passed else "출력 검증 실패",
    ]
    val_messages.extend(validation.checks[:4])
    for err in validation.errors[:4]:
        val_messages.append(f"error: {err}")

    def _populate_val(t) -> None:
        t.decision = {
            "passed": validation.passed,
            "checks": validation.checks[:8],
            "errors": validation.errors[:8],
        }
        t.summary = "출력 검증 완료" if validation.passed else "출력 검증 실패"
        if not validation.passed:
            t.status = "warning"
            t.warnings.extend(validation.errors[:4])

    bridge.emit_step(
        step_id="validator",
        agent="Validator",
        legacy_status="검증 완료",
        legacy_messages=val_messages,
        populate=_populate_val if bridge.enable_structured else None,
    )

    return _finish(
        pipeline_status="completed",
        decision=decision,
        validation=validation,
    )


def run_enhanced_pipeline(
    user_input: UserInput,
    drugs: List[Drug],
    symptoms: List[Symptom],
    top_k: int = 3,
    config: Optional[AppConfig] = None,
    *,
    context_text: str = "",
    on_step: Optional[Callable[[AgentStep], None]] = None,
) -> str:
    """파이프라인 실행 후 사람이 읽을 수 있는 텍스트 응답을 반환합니다."""
    cfg = config or load_config()
    result = run_enhanced_pipeline_core(
        user_input,
        drugs,
        symptoms,
        top_k=top_k,
        config=cfg,
        context_text=context_text,
        on_step=on_step,
    )
    return _format_enhanced_response(
        user_input,
        result.decision,
        result.validation,
        result.trace_steps,
        use_chroma=cfg.use_chroma,
    )
