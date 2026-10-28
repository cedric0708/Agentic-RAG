from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from agents.models import ChromaEvidence, Drug, RetrievalResult, UserInput

if TYPE_CHECKING:
    from agents.config import AppConfig
    from agents.integrations.context import DeIntegrationContext

from agents.integrations.mapping import build_symptom_queries, drug_from_de_row
from agents.integrations.types import DataBackend, VectorSearchOutcome


def _normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def _extract_treats_from_chunks(drug: Drug) -> Set[str]:
    treats: Set[str] = set()
    for chunk in drug.child_chunks:
        chunk_treats = chunk.metadata.get("treats", [])
        for item in chunk_treats:
            if item:
                treats.add(str(item).strip().lower())
    return treats


def score_drug(user_input: UserInput, drug: Drug) -> RetrievalResult:
    score = 0.0
    matched_symptoms: List[str] = []
    reasons: List[str] = []

    indications_text = _normalize_text(drug.indications)
    parent_text = _normalize_text(drug.parent_text)
    treats_set = _extract_treats_from_chunks(drug)

    for symptom in user_input.symptoms:
        symptom_norm = _normalize_text(symptom)
        if not symptom_norm:
            continue

        matched = False

        if symptom_norm in indications_text:
            score += 3.0
            matched = True
            reasons.append(f"indications에 '{symptom}' 포함 (+3)")

        if symptom_norm in parent_text:
            score += 1.5
            matched = True
            reasons.append(f"parent_text에 '{symptom}' 포함 (+1.5)")

        if symptom_norm in treats_set:
            score += 2.0
            matched = True
            reasons.append(f"child metadata treats에 '{symptom}' 포함 (+2)")

        if matched:
            matched_symptoms.append(symptom)

    if drug.category:
        score += 0.5
        reasons.append("category 정보 존재 (+0.5)")

    if drug.updated_date:
        score += 0.5
        reasons.append("updated_date 정보 존재 (+0.5)")

    if drug.warnings:
        score -= 0.3
        reasons.append("주의사항 존재 (-0.3)")

    return RetrievalResult(
        drug=drug,
        score=score,
        matched_symptoms=matched_symptoms,
        reasons=reasons,
    )


def _build_chroma_query(user_input: UserInput, user_text: str = "") -> str:
    parts = [s.strip() for s in user_input.symptoms if s.strip()]
    norm = _normalize_text(user_text)
    for cue in ("머리", "목", "배", "복", "허리", "속", "인후"):
        if cue in norm and cue not in " ".join(parts):
            parts.append(cue)
    return " ".join(parts)


def _chroma_drug_signals(hits) -> tuple[Dict[str, float], Dict[str, List[ChromaEvidence]]]:
    scores: Dict[str, float] = {}
    evidence_map: Dict[str, List[ChromaEvidence]] = {}

    for hit in hits:
        meta = hit.metadata or {}
        drug_id = None
        data_type = meta.get("data_type")

        if data_type == "drug":
            drug_id = str(meta.get("entity_id") or "")
        elif data_type == "mapping":
            drug_id = str(meta.get("drug_id") or "")

        if not drug_id:
            continue

        relevance = max(0.0, 1.0 - float(hit.distance))
        scores[drug_id] = max(scores.get(drug_id, 0.0), relevance)

        evidence = ChromaEvidence(
            chunk_id=hit.id,
            document=hit.document,
            metadata=meta,
            distance=float(hit.distance),
            relevance=relevance,
        )
        evidence_map.setdefault(drug_id, []).append(evidence)

    return scores, evidence_map


def _search_chroma(
    user_input: UserInput,
    chroma_top_n: int,
) -> tuple[Dict[str, float], Dict[str, List[ChromaEvidence]]]:
    """ChromaDB 검색. 실패 시 빈 결과 반환."""
    query = _build_chroma_query(user_input)
    if not query:
        return {}, {}

    try:
        from agents.chroma_retriever import search_medical_knowledge

        hits = search_medical_knowledge(query, n_results=chroma_top_n)
        return _chroma_drug_signals(hits)
    except Exception:
        return {}, {}


def _select_candidate_ids(
    drugs: List[Drug],
    drug_by_id: Dict[str, Drug],
    chroma_scores: Dict[str, float],
    user_input: UserInput,
    *,
    large_catalog_threshold: int = 100,
) -> Set[str]:
    if len(drugs) <= large_catalog_threshold:
        return set(drug_by_id) | set(chroma_scores)

    candidate_ids: Set[str] = set(chroma_scores)
    for drug in drugs:
        base = score_drug(user_input, drug)
        if base.matched_symptoms:
            candidate_ids.add(drug.drug_id)
    return candidate_ids


def retrieve_top_k(
    user_input: UserInput,
    drugs: List[Drug],
    top_k: int = 3,
    *,
    use_chroma: bool = True,
    chroma_top_n: int = 10,
    chroma_weight: float = 5.0,
    min_score: float = 0.0,
    large_catalog_threshold: int = 100,
) -> List[RetrievalResult]:
    """규칙 기반 점수 + ChromaDB 유사도 boost 하이브리드 검색."""
    drug_by_id = {drug.drug_id: drug for drug in drugs}

    chroma_scores: Dict[str, float] = {}
    chroma_evidence: Dict[str, List[ChromaEvidence]] = {}

    if use_chroma:
        chroma_top_n = max(chroma_top_n, top_k * 5)
        chroma_scores, chroma_evidence = _search_chroma(user_input, chroma_top_n)

    scored_results: List[RetrievalResult] = []
    candidate_ids = _select_candidate_ids(
        drugs,
        drug_by_id,
        chroma_scores,
        user_input,
        large_catalog_threshold=large_catalog_threshold,
    )

    for drug_id in candidate_ids:
        drug = drug_by_id.get(drug_id)
        if drug is None:
            continue

        base = score_drug(user_input, drug)
        boost = chroma_scores.get(drug_id, 0.0) * chroma_weight
        total_score = base.score + boost

        reasons = list(base.reasons)
        if boost > 0:
            reasons.append(f"ChromaDB 유사도 boost +{boost:.2f}")

        evidence = chroma_evidence.get(drug_id, [])
        if total_score <= min_score and not evidence:
            continue

        scored_results.append(
            RetrievalResult(
                drug=drug,
                score=total_score if total_score > 0 else boost,
                matched_symptoms=base.matched_symptoms,
                reasons=reasons,
                chroma_evidence=evidence,
            )
        )

    scored_results.sort(key=lambda item: item.score, reverse=True)
    return scored_results[:top_k]


def _retrieve_from_postgresql(
    user_input: UserInput,
    drugs: List[Drug],
    top_k: int,
    *,
    de_ctx: "DeIntegrationContext",
    user_text: str,
    de_vector_weight: float,
    multiplier: int,
) -> Tuple[List[RetrievalResult], VectorSearchOutcome]:
    queries = build_symptom_queries(user_input, user_text)
    outcome = de_ctx.vector.search_drugs(queries, top_k=max(top_k * multiplier, top_k))
    if not outcome.ok or not outcome.rows:
        return [], outcome

    results: List[RetrievalResult] = []
    for row in outcome.rows:
        drug, match_kind = drug_from_de_row(row, drugs)
        de_score = float(row.get("score") or 0.0)
        rule = score_drug(user_input, drug)
        total = de_score * de_vector_weight + rule.score * 0.25
        reasons = [
            f"PostgreSQL/pgvector score={de_score:.3f} (×{de_vector_weight})",
            f"retrieval_source={outcome.backend.value}",
        ]
        if match_kind == "stub":
            reasons.append("catalog 미매칭 stub — legacy 상세 정보 없음")
        results.append(
            RetrievalResult(
                drug=drug,
                score=total,
                matched_symptoms=rule.matched_symptoms or list(user_input.symptoms),
                reasons=reasons,
                de_score=de_score,
                retrieval_source=outcome.backend.value,
                de_document_snippet=str(row.get("text") or "")[:200] or None,
                catalog_match=match_kind,
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k], outcome


def retrieve_candidates(
    user_input: UserInput,
    drugs: List[Drug],
    top_k: int = 3,
    *,
    config: "AppConfig",
    de_ctx: Optional["DeIntegrationContext"] = None,
    user_text: str = "",
    use_chroma: bool = True,
    chroma_top_n: int = 10,
    chroma_weight: float = 5.0,
    min_score: float = 0.0,
) -> Tuple[List[RetrievalResult], DataBackend, List[str]]:
    """
    DE PostgreSQL 우선 검색 + (실패 시) legacy JSON/Chroma fallback.
    """
    notes: List[str] = []
    de_results: List[RetrievalResult] = []
    retrieval_backend = DataBackend.LEGACY_JSON

    use_vector = (
        de_ctx is not None
        and config.use_de_vector
        and de_ctx.vector.backend != DataBackend.DISABLED
    )

    if use_vector:
        de_results, outcome = _retrieve_from_postgresql(
            user_input,
            drugs,
            top_k,
            de_ctx=de_ctx,
            user_text=user_text,
            de_vector_weight=config.de_vector_score_weight,
            multiplier=config.de_retrieval_top_k_multiplier,
        )
        if outcome.ok and de_results:
            retrieval_backend = outcome.backend
            notes.append(f"Retrieval: {outcome.backend.value} hits={len(de_results)}")
        elif outcome.attempted_postgresql:
            notes.append(
                f"PostgreSQL 검색 실패({outcome.backend.value}): "
                f"{(outcome.error or 'no rows')[:120]}"
            )
            if not config.de_fallback_legacy:
                notes.append("DE_FALLBACK_LEGACY=false — legacy 검색 생략")
                return [], DataBackend.UNAVAILABLE, notes

    if de_results:
        if not config.de_fallback_legacy:
            return de_results, retrieval_backend, notes
        legacy = retrieve_top_k(
            user_input,
            drugs,
            top_k=top_k,
            use_chroma=use_chroma and config.use_chroma,
            chroma_top_n=chroma_top_n,
            chroma_weight=chroma_weight,
            min_score=min_score,
        )
        seen = {r.drug.drug_id for r in de_results}
        merged = list(de_results)
        for item in legacy:
            if item.drug.drug_id in seen:
                continue
            item.retrieval_source = DataBackend.LEGACY_JSON.value
            merged.append(item)
            if len(merged) >= top_k:
                break
        if legacy:
            retrieval_backend = DataBackend.HYBRID
            notes.append(f"hybrid: DE {len(de_results)} + legacy 보충")
        return merged[:top_k], retrieval_backend, notes

    legacy_only = retrieve_top_k(
        user_input,
        drugs,
        top_k=top_k,
        use_chroma=use_chroma and config.use_chroma,
        chroma_top_n=chroma_top_n,
        chroma_weight=chroma_weight,
        min_score=min_score,
    )
    if use_vector:
        retrieval_backend = DataBackend.LEGACY_JSON
        notes.append("Retrieval fallback=legacy_json (DE 미사용 또는 실패)")
    for item in legacy_only:
        item.retrieval_source = DataBackend.LEGACY_JSON.value
    return legacy_only, retrieval_backend, notes


def retrieve_symptom_context(
    user_input: UserInput,
    top_n: int = 3,
    *,
    user_text: str = "",
) -> List[ChromaEvidence]:
    """증상 안내/경고 청크를 ChromaDB에서 가져옵니다."""
    from agents.symptom_context_filter import filter_symptom_context

    query = _build_chroma_query(user_input, user_text)
    if not query:
        return []

    try:
        from agents.chroma_retriever import search_medical_knowledge

        try:
            hits = search_medical_knowledge(
                query,
                n_results=top_n * 4,
                where={"data_type": "symptom"},
            )
        except Exception:
            hits = search_medical_knowledge(query, n_results=top_n * 4)
    except Exception:
        return []

    context: List[ChromaEvidence] = []
    for hit in hits:
        if hit.metadata.get("data_type") not in (None, "symptom"):
            continue
        context.append(
            ChromaEvidence(
                chunk_id=hit.id,
                document=hit.document,
                metadata=hit.metadata,
                distance=float(hit.distance),
                relevance=max(0.0, 1.0 - float(hit.distance)),
            )
        )

    return filter_symptom_context(
        context,
        user_input,
        user_text=user_text,
        top_n=top_n,
    )
