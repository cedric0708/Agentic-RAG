from __future__ import annotations



from typing import List, Optional, Tuple



from agents.integrations.graph_client import DrugGraphClient

from agents.integrations.mapping import resolve_current_item_seqs, user_is_pregnant

from agents.integrations.types import AssessOutcome, DataBackend

from agents.models import Drug, UserInput



# 추천 제외(blocked). UNKNOWN/NOT_FOUND는 «병용금기»가 아니라 «판정 불가».

VERDICT_BLOCK = frozenset({"BAN", "NOT_FOUND", "UNKNOWN"})



# 병용금기로 표현해도 되는 verdict (실제 DUR 상호작용 판정)

VERDICT_CONTRAINDICATION = frozenset({"BAN"})





def _append_finding_warnings(verdict: str, findings: list, warnings: List[str]) -> None:

    for f in findings or []:

        ftype = str(f.get("type") or "")

        msg = f.get("message") or ftype or "Neo4j finding"

        if verdict in VERDICT_CONTRAINDICATION and ftype in (

            "contraindication",

            "duplicate_ingredient",

            "effect_duplicate",

        ):

            warnings.append(f"[식약처 DUR 병용·중복] {msg}")

        elif verdict == "CAUTION":

            warnings.append(f"[DUR 주의] {msg}")

        elif verdict not in VERDICT_BLOCK:

            warnings.append(f"[Neo4j] {msg}")





def _verdict_user_message(verdict: str) -> str:

    if verdict == "BAN":

        return (

            "Neo4j assess: 식약처 DUR 기준 병용금기·금기에 해당합니다. "

            "이 후보는 추천에서 제외됩니다(안전하다고 판단하지 않음)."

        )

    if verdict == "UNKNOWN":

        return (

            "Neo4j assess: 성분 정보가 부족해 금기 여부를 확인하지 못했습니다. "

            "병용금기로 확정하지 않으며, 추천에서 제외합니다. 약사 확인이 필요합니다."

        )

    if verdict == "NOT_FOUND":

        return (

            "Neo4j assess: 그래프에서 해당 품목을 찾지 못했습니다. "

            "병용금기로 확정하지 않으며, 추천에서 제외합니다."

        )

    if verdict == "CAUTION":

        return (

            "Neo4j assess: DUR 기준 주의(성분 중복·효능군 중복 등)가 있습니다. "

            "확정적 안전 추천이 아니며 약사 상담을 권장합니다."

        )

    if verdict == "OK":

        return (

            "Neo4j assess: DUR 기준 확인된 금기·주의가 없습니다. "

            "의학적 안전·적합성을 보장하지 않으며, 복용 전 약사·의사 상담이 필요합니다."

        )

    return f"Neo4j assess verdict={verdict}"





def merge_assess_into_contraindication(

    base_blocked: bool,

    base_penalty: float,

    base_warnings: List[str],

    base_risk: str,

    outcome: AssessOutcome,

    *,

    fallback_legacy: bool,

) -> Tuple[bool, float, List[str], str, DataBackend, List[str], bool]:

    """

    Neo4j assess 결과를 legacy contraindication 결과에 병합.



    Returns:

        blocked, penalty, warnings, risk, safety_backend, notes, de_safety_verified

    """

    notes: List[str] = []

    blocked = base_blocked

    penalty = base_penalty

    warnings = list(base_warnings)

    risk = base_risk

    safety_backend = DataBackend.LEGACY_JSON

    de_safety_verified = True



    if not outcome.attempted_neo4j:

        return blocked, penalty, warnings, risk, safety_backend, notes, de_safety_verified



    if not outcome.ok or not outcome.payload:

        safety_backend = DataBackend.UNAVAILABLE

        de_safety_verified = False

        if fallback_legacy:

            notes.append("Neo4j assess 미수행 — legacy JSON 안전성만 적용(미검증)")

            warnings.append(

                "DE(Neo4j) 금기 판정을 수행하지 못했습니다. "

                "legacy 데이터만 반영되었으며 DUR 기준 미검증입니다."

            )

        else:

            blocked = True

            penalty = max(penalty, 8.0)

            risk = "high"

            warnings.append(

                "Neo4j 연결 실패 — DE 우선 모드에서 후보를 추천할 수 없습니다."

            )

        return blocked, penalty, warnings, risk, safety_backend, notes, de_safety_verified



    payload = outcome.payload

    verdict = str(payload.get("verdict") or "UNKNOWN").upper()

    safety_backend = outcome.backend

    findings = payload.get("findings") or []



    _append_finding_warnings(verdict, findings, warnings)



    for n in payload.get("notes") or []:

        notes.append(str(n))



    warnings.append(_verdict_user_message(verdict))



    if verdict in VERDICT_BLOCK:

        blocked = True

        penalty = max(penalty, 10.0 if verdict == "BAN" else 7.0)

        risk = "high"

    elif verdict == "CAUTION":

        penalty = max(penalty, 3.0)

        if risk == "low":

            risk = "medium"

    elif verdict == "OK":

        de_safety_verified = True



    return blocked, penalty, warnings, risk, safety_backend, notes, de_safety_verified





def run_neo4j_assess(

    graph: DrugGraphClient,

    user_input: UserInput,

    candidate: Drug,

    drugs: List[Drug],

) -> AssessOutcome:

    find_fn = graph.find_drug if graph.backend not in (DataBackend.DISABLED,) else None

    current, _ = resolve_current_item_seqs(

        user_input,

        drugs,

        find_drug_fn=find_fn,

    )

    age = float(user_input.age) if user_input.age is not None else None

    pregnant = user_is_pregnant(user_input)

    return graph.assess(

        current,

        candidate.drug_id,

        age=age,

        pregnant=pregnant,

        with_graph=False,

    )
