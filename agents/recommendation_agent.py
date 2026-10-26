from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from agents.reranker import RerankedResult


@dataclass
class RecommendationDecision:
    recommended: List[RerankedResult] = field(default_factory=list)
    rejected: List[RerankedResult] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)


def _looks_pediatric_only(drug) -> bool:
    dose = drug.dosage or ""
    name = drug.name_ko or ""
    if "시럽" in name and "성인" not in dose:
        return True
    if "만 12세 이하" in dose and "성인" not in dose:
        return True
    if "소아" in dose and "성인" not in dose:
        return True
    return False


def _is_symptom_relevant(item: RerankedResult) -> bool:
    retrieval = item.retrieval
    if retrieval.de_score is not None and retrieval.de_score >= 0.35:
        return True
    if retrieval.retrieval_source in {"postgresql", "mock", "hybrid"} and retrieval.score >= 1.0:
        return True
    if retrieval.matched_symptoms:
        return True
    if retrieval.score >= 1.5:
        return True
    if retrieval.chroma_evidence:
        max_rel = max(ev.relevance for ev in retrieval.chroma_evidence)
        if max_rel >= 0.40 and retrieval.score >= 1.0:
            return True
    indications = (retrieval.drug.indications or "").lower()
    for symptom in retrieval.matched_symptoms:
        if symptom.lower() in indications:
            return True
    return False


def decide_recommendations(
    reranked_results: List[RerankedResult],
    max_recommendations: int = 3,
) -> RecommendationDecision:
    """재정렬 결과를 추천/비추천으로 나눈다."""

    recommended: List[RerankedResult] = []
    rejected: List[RerankedResult] = []
    messages: List[str] = []

    ordered = sorted(
        reranked_results,
        key=lambda item: (
            1 if _looks_pediatric_only(item.retrieval.drug) else 0,
            -item.final_score,
        ),
    )

    stub_rejected = 0
    unverified_rejected = 0

    for item in ordered:
        if item.blocked:
            rejected.append(item)
            continue
        if item.needs_extra_verification:
            if item.retrieval.catalog_match == "stub":
                stub_rejected += 1
            if any("de_safety_verified=false" in r for r in item.rerank_reasons):
                unverified_rejected += 1
            rejected.append(item)
            continue
        if not _is_symptom_relevant(item):
            rejected.append(item)
            continue
        if item.final_score <= 0 and item.risk_level == "high":
            rejected.append(item)
            continue
        if len(recommended) < max_recommendations:
            recommended.append(item)
        else:
            rejected.append(item)

    if recommended:
        top = recommended[0].retrieval.drug
        messages.append(f"최종 1순위 후보는 {top.name_ko}입니다.")
        if recommended[0].risk_level in {"medium", "high"}:
            messages.append(
                f"{top.name_ko}은(는) 증상과는 맞지만 주의가 필요해 주의 깊게 안내합니다."
            )
    else:
        messages.append(
            "안전성·관련성 기준을 통과한 추천 후보가 없습니다. 약사/의료진 상담이 필요합니다."
        )

    if stub_rejected:
        messages.append(
            f"로컬 catalog에 없는 DE 검색 결과(stub) {stub_rejected}건은 "
            "성분·용법·금기 미확인으로 확정 추천하지 않았습니다."
        )
    if unverified_rejected:
        messages.append(
            f"Neo4j(DUR) 안전성 미검증 후보 {unverified_rejected}건은 "
            "안전하다고 판단하지 않고 추천에서 제외했습니다."
        )
    if rejected:
        messages.append(f"관련성 부족 또는 안전성 이유로 제외된 후보 {len(rejected)}개가 있습니다.")

    return RecommendationDecision(
        recommended=recommended,
        rejected=rejected,
        messages=messages,
    )
