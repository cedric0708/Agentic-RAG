from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from agents.integrations.graph_client import DrugGraphClient
from agents.integrations.safety_de import merge_assess_into_contraindication, run_neo4j_assess
from agents.integrations.types import DataBackend
from agents.models import Drug, UserInput
from agents.safety import evaluate_drug_safety


def _norm(text: str) -> str:
    return (text or "").strip().lower()


@dataclass
class ContraindicationCheck:
    drug_id: str
    drug_name: str
    risk_level: str
    warnings: List[str] = field(default_factory=list)
    blocked: bool = False
    penalty: float = 0.0
    safety_backend: str = DataBackend.LEGACY_JSON.value
    neo4j_verdict: str | None = None
    de_safety_verified: bool = True


def _current_drug_lookup(drugs: List[Drug]) -> Dict[str, Drug]:
    lookup: Dict[str, Drug] = {}
    for drug in drugs:
        lookup[_norm(drug.drug_id)] = drug
        lookup[_norm(drug.name_ko)] = drug
        if drug.name_en:
            lookup[_norm(drug.name_en)] = drug
    return lookup


def _extract_current_ingredients(user_input: UserInput, drugs: List[Drug]) -> Set[str]:
    lookup = _current_drug_lookup(drugs)
    ingredients: Set[str] = set()
    for key in user_input.current_drug_ids + user_input.current_drug_names:
        drug = lookup.get(_norm(key))
        if drug:
            for ingredient in drug.ingredient:
                ingredients.add(_norm(ingredient))
    return ingredients


def check_drug_contraindication(
    user_input: UserInput,
    candidate: Drug,
    all_drugs: List[Drug],
    *,
    graph_client: DrugGraphClient | None = None,
    fallback_legacy: bool = True,
) -> ContraindicationCheck:
    """후보 약 하나에 대해 병용금기/중복성분/사용자 상태 위험을 검사한다."""

    base_safety = evaluate_drug_safety(user_input, candidate)
    warnings = base_safety.interaction_warnings + base_safety.contraindication_warnings
    penalty = 0.0
    blocked = False

    if base_safety.interaction_warnings:
        penalty += 4.0
        blocked = True
    if base_safety.contraindication_warnings:
        penalty += 3.0

    current_ingredients = _extract_current_ingredients(user_input, all_drugs)
    candidate_ingredients = {_norm(item) for item in candidate.ingredient}
    duplicated = sorted(current_ingredients & candidate_ingredients)
    if duplicated:
        warnings.append(f"현재 복용약과 중복 성분 감지: {', '.join(duplicated)}")
        penalty += 3.0

    forbidden: set[str] = set()
    for chunk in candidate.child_chunks:
        for item in chunk.metadata.get("forbidden_conditions", []) or []:
            forbidden.add(_norm(str(item)))

    for condition in user_input.conditions:
        condition_norm = _norm(condition)
        if not condition_norm or condition_norm == _norm("없음"):
            continue
        if condition_norm == _norm("음주") and _norm("음주") in forbidden:
            warnings.append(
                f"{candidate.name_ko}: 음주 시 복용 전 약사 상담을 권장합니다."
            )
            penalty += 0.5
            continue
        if any(condition_norm in f or f in condition_norm for f in forbidden if f):
            penalty += 2.0
            warnings.append(
                f"사용자 상태 '{condition}' 관련 금기/주의 성분이 있습니다."
            )

    if penalty >= 4.0:
        risk_level = "high"
    elif penalty >= 2.0:
        risk_level = "medium"
    else:
        risk_level = "low"

    safety_backend = DataBackend.LEGACY_JSON
    neo4j_verdict: str | None = None
    de_safety_verified = True

    if graph_client is not None and graph_client.backend != DataBackend.DISABLED:
        assess_outcome = run_neo4j_assess(graph_client, user_input, candidate, all_drugs)
        (
            blocked,
            penalty,
            warnings,
            risk_level,
            merged_backend,
            _notes,
            de_safety_verified,
        ) = merge_assess_into_contraindication(
            blocked,
            penalty,
            warnings,
            risk_level,
            assess_outcome,
            fallback_legacy=fallback_legacy,
        )
        if assess_outcome.payload:
            neo4j_verdict = str(assess_outcome.payload.get("verdict") or "")
        if merged_backend not in (DataBackend.LEGACY_JSON, DataBackend.DISABLED):
            safety_backend = merged_backend
        elif assess_outcome.attempted_neo4j and not assess_outcome.ok:
            safety_backend = DataBackend.UNAVAILABLE

    return ContraindicationCheck(
        drug_id=candidate.drug_id,
        drug_name=candidate.name_ko,
        risk_level=risk_level,
        warnings=warnings,
        blocked=blocked,
        penalty=penalty,
        safety_backend=safety_backend.value,
        neo4j_verdict=neo4j_verdict,
        de_safety_verified=de_safety_verified,
    )


def check_all_contraindications(user_input: UserInput, candidates: List[Drug], all_drugs: List[Drug]) -> List[ContraindicationCheck]:
    return [check_drug_contraindication(user_input, drug, all_drugs) for drug in candidates]
