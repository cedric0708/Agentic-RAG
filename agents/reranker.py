from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from agents.contraindication import check_drug_contraindication
from agents.integrations.graph_client import DrugGraphClient
from agents.models import Drug, RetrievalResult, UserInput


@dataclass
class RerankedResult:
    retrieval: RetrievalResult
    final_score: float
    safety_penalty: float = 0.0
    confidence: float = 0.0
    blocked: bool = False
    risk_level: str = "low"
    rerank_reasons: List[str] = field(default_factory=list)
    needs_extra_verification: bool = False


def _confidence_from_score(score: float, matched_count: int, penalty: float) -> float:
    raw = (score * 0.12) + (matched_count * 0.18) - (penalty * 0.08)
    return max(0.0, min(1.0, raw))


def rerank_results(
    user_input: UserInput,
    results: List[RetrievalResult],
    all_drugs: List[Drug],
    *,
    graph_client: DrugGraphClient | None = None,
    de_fallback_legacy: bool = True,
) -> List[RerankedResult]:
    """검색 결과를 안전성 penalty와 매칭 근거 기반으로 재정렬한다."""

    reranked: List[RerankedResult] = []
    for result in results:
        risk = check_drug_contraindication(
            user_input,
            result.drug,
            all_drugs,
            graph_client=graph_client,
            fallback_legacy=de_fallback_legacy,
        )
        final_score = result.score - risk.penalty
        reasons = list(result.reasons)

        if risk.penalty:
            reasons.append(f"안전성 penalty -{risk.penalty:.1f} ({risk.risk_level})")
        else:
            reasons.append("병용금기/상태 기반 penalty 없음")
        reasons.append(f"safety_backend={risk.safety_backend}")
        if risk.neo4j_verdict:
            reasons.append(f"neo4j_verdict={risk.neo4j_verdict}")
        if risk.blocked:
            reasons.append("blocked=true (추천 제외)")
        if not risk.de_safety_verified:
            reasons.append("de_safety_verified=false (Neo4j 미검증)")
        if result.catalog_match == "stub":
            reasons.append("catalog_match=stub (로컬 catalog·용법·성분 미확인)")

        needs_extra = result.catalog_match == "stub" or not risk.de_safety_verified

        if result.matched_symptoms:
            reasons.append(f"매칭 증상 {len(result.matched_symptoms)}개 반영")

        confidence = _confidence_from_score(result.score, len(result.matched_symptoms), risk.penalty)
        reranked.append(
            RerankedResult(
                retrieval=result,
                final_score=final_score,
                safety_penalty=risk.penalty,
                confidence=confidence,
                blocked=risk.blocked,
                risk_level=risk.risk_level,
                rerank_reasons=reasons,
                needs_extra_verification=needs_extra,
            )
        )

    reranked.sort(key=lambda item: (item.blocked, -item.final_score))
    return reranked
