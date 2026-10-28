"""Drug Interaction Network Graph — Neo4j 기반 JSON (WEB 시각화용)."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from agents.integrations.graph_client import DrugGraphClient
from agents.integrations.mapping import (
    item_seq_from_drug_id,
    lookup_drug_by_id,
    lookup_drug_by_name,
    normalize_key,
)
from agents.integrations.types import DataBackend, DataProvenance, provenance_to_dict
from agents.models import Drug

MAX_DRUGS = 10
INTERACTION_FINDING_TYPES = frozenset(
    {"contraindication", "duplicate_ingredient", "effect_duplicate"}
)
DISCLAIMER = (
    "본 그래프는 식약처 DUR·Neo4j에 기록된 관계와 AI가 조회한 findings를 표시합니다. "
    "상호작용 엣지가 없다고 복용 안전을 의미하지 않으며, 의학적 안전을 보장하지 않습니다."
)


@dataclass
class MappedDrug:
    input_kind: str  # id | name
    input_value: str
    item_seq: Optional[str] = None
    label: str = ""
    mapping_status: str = "NOT_MAPPED"  # RESOLVED | AMBIGUOUS | NOT_MAPPED | ...
    graph_status: str = "NOT_MAPPED"
    candidates: List[dict] = field(default_factory=list)
    catalog_item_seq: Optional[str] = None
    neo4j_item_seq: Optional[str] = None


def _drug_node_id(item_seq: str) -> str:
    return f"drug:{item_seq}"


def _ingr_node_id(code: str) -> str:
    return f"ingr:{code}"


def _unmapped_node_id(kind: str, value: str) -> str:
    h = hashlib.sha256(f"{kind}:{value}".encode()).hexdigest()[:12]
    return f"drug:unmapped:{h}"


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _canonical_ingredient_codes(
    code_a: Optional[str], code_b: Optional[str]
) -> Tuple[str, str]:
    a = str(code_a or "").strip()
    b = str(code_b or "").strip()
    if not a and not b:
        return ("", "")
    if not a:
        return (b, b)
    if not b:
        return (a, a)
    return (a, b) if a <= b else (b, a)


def _finding_relation_key(f: dict) -> Tuple[str, str, str]:
    ftype = str(f.get("type") or "")
    ca, cb = _canonical_ingredient_codes(f.get("current_code"), f.get("candidate_code"))
    return (ftype, ca, cb)


def _evidence_item_from_finding(f: dict) -> Dict[str, Any]:
    return {
        "message": f.get("message") or "",
        "evidence_source": f.get("source"),
        "source_url": f.get("source_url"),
    }


def _merge_finding_evidence(existing: dict, incoming: dict) -> None:
    items: List[dict] = existing.setdefault("_evidence_items", [])
    seen = {
        (
            str(it.get("message") or ""),
            str(it.get("evidence_source") or ""),
            str(it.get("source_url") or ""),
        )
        for it in items
    }
    cand = _evidence_item_from_finding(incoming)
    key = (
        cand["message"],
        str(cand.get("evidence_source") or ""),
        str(cand.get("source_url") or ""),
    )
    if key not in seen:
        seen.add(key)
        items.append(cand)
    level_rank = {"ban": 2, "caution": 1}
    cur = str(existing.get("level") or "caution")
    new = str(incoming.get("level") or "caution")
    if level_rank.get(new, 0) > level_rank.get(cur, 0):
        existing["level"] = new


def _merge_bidirectional_findings(findings: List[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, str, str], dict] = {}
    for f in findings:
        key = _finding_relation_key(f)
        ca, cb = key[1], key[2]
        if key not in grouped:
            copy = dict(f)
            copy["current_code"] = ca or f.get("current_code")
            copy["candidate_code"] = cb or f.get("candidate_code")
            copy["_evidence_items"] = [_evidence_item_from_finding(f)]
            grouped[key] = copy
        else:
            _merge_finding_evidence(grouped[key], f)
    return list(grouped.values())


def _edge_evidence_fields(f: dict) -> Dict[str, Any]:
    items: List[dict] = f.get("_evidence_items") or [_evidence_item_from_finding(f)]
    primary = items[0]
    fields: Dict[str, Any] = {
        "message": primary.get("message") or "",
        "evidence_source": primary.get("evidence_source"),
        "source_url": primary.get("source_url"),
    }
    if len(items) > 1:
        fields["evidence_items"] = items
    return fields


def _edge_id(*parts: str) -> str:
    raw = "|".join(parts)
    return "e:" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def _verdict_from_level(level: str, finding_type: str) -> str:
    if level == "ban" or finding_type == "contraindication":
        return "BAN"
    if level == "caution":
        return "CAUTION"
    return "CAUTION"


def _sanitize_error(msg: str) -> str:
    msg = re.sub(r"(?i)(password|neo4j(\+s)?://[^@\s]+:[^@\s]+@)[^\s]*", "[REDACTED]", msg or "")
    return msg[:300]


def normalize_drug_inputs(
    drug_ids: List[str],
    drug_names: List[str],
    *,
    catalog: List[Drug],
    graph: DrugGraphClient,
) -> Tuple[List[MappedDrug], List[str]]:
    """입력 정규화·매핑 (가짜 item_seq 부여 금지)."""
    warnings: List[str] = []
    mapped: List[MappedDrug] = []
    seen_seq: Set[str] = set()
    seen_name_keys: Set[str] = set()

    backend_unavailable = graph.backend in (DataBackend.DISABLED, DataBackend.UNAVAILABLE)

    for raw_id in drug_ids or []:
        seq = item_seq_from_drug_id(raw_id)
        if not seq:
            continue
        if seq in seen_seq:
            continue
        seen_seq.add(seq)
        entry = MappedDrug(input_kind="id", input_value=raw_id, item_seq=seq, label=seq)
        catalog_drug = lookup_drug_by_id(catalog, seq)
        if catalog_drug:
            entry.catalog_item_seq = seq
            entry.label = catalog_drug.name_ko
        if backend_unavailable:
            entry.mapping_status = "UNAVAILABLE"
            entry.graph_status = "UNAVAILABLE"
        else:
            st = graph.check_status(seq)
            entry.graph_status = st.get("status") or "NOT_FOUND"
            drug = st.get("drug") or {}
            if drug.get("name"):
                entry.label = drug["name"]
            if entry.graph_status == "NOT_FOUND":
                entry.mapping_status = "NOT_FOUND"
                entry.item_seq = seq
            elif entry.graph_status == "UNKNOWN":
                entry.mapping_status = "RESOLVED"
                entry.item_seq = seq
            else:
                entry.mapping_status = "RESOLVED"
                entry.neo4j_item_seq = seq
        mapped.append(entry)

    for name in drug_names or []:
        name = (name or "").strip()
        if not name:
            continue
        nkey = normalize_key(name)
        if nkey in seen_name_keys:
            continue
        seen_name_keys.add(nkey)
        entry = MappedDrug(input_kind="name", input_value=name, label=name)
        catalog_drug = lookup_drug_by_name(catalog, name)
        if catalog_drug:
            entry.catalog_item_seq = catalog_drug.drug_id
            entry.label = catalog_drug.name_ko
            if catalog_drug.drug_id in seen_seq:
                warnings.append(f"이름 '{name}'은(는) 이미 ID 입력과 중복되어 생략됩니다.")
                continue

        if backend_unavailable:
            entry.mapping_status = "UNAVAILABLE"
            entry.graph_status = "UNAVAILABLE"
            mapped.append(entry)
            continue

        hits = graph.find_drug(name, limit=5)
        entry.candidates = list(hits)
        exact = [h for h in hits if normalize_key(h.get("name") or "") == nkey]
        neo_seq = None
        if len(exact) == 1:
            neo_seq = str(exact[0].get("item_seq") or "")
        elif len(hits) == 1:
            neo_seq = str(hits[0].get("item_seq") or "")
        elif len(hits) > 1 and not exact:
            entry.mapping_status = "AMBIGUOUS"
            entry.graph_status = "AMBIGUOUS"
            mapped.append(entry)
            continue
        elif not hits:
            entry.mapping_status = "NOT_MAPPED"
            entry.graph_status = "NOT_MAPPED"
            mapped.append(entry)
            continue
        else:
            entry.mapping_status = "AMBIGUOUS"
            entry.graph_status = "AMBIGUOUS"
            mapped.append(entry)
            continue

        if entry.catalog_item_seq and neo_seq and entry.catalog_item_seq != neo_seq:
            entry.mapping_status = "AMBIGUOUS"
            entry.graph_status = "AMBIGUOUS"
            entry.neo4j_item_seq = neo_seq
            warnings.append(
                f"이름 '{name}': catalog({entry.catalog_item_seq}) vs Neo4j({neo_seq}) 불일치"
            )
            mapped.append(entry)
            continue

        if neo_seq in seen_seq:
            warnings.append(f"이름 '{name}' → {neo_seq} 는 이미 목록에 있습니다.")
            continue
        seen_seq.add(neo_seq)
        entry.item_seq = neo_seq
        entry.neo4j_item_seq = neo_seq
        entry.mapping_status = "RESOLVED"
        st = graph.check_status(neo_seq)
        entry.graph_status = st.get("status") or "NOT_FOUND"
        if entry.graph_status == "NOT_FOUND":
            entry.mapping_status = "NOT_FOUND"
        if (st.get("drug") or {}).get("name"):
            entry.label = st["drug"]["name"]
        mapped.append(entry)

    return mapped, warnings


def build_interaction_graph(
    *,
    drug_ids: List[str],
    drug_names: List[str],
    catalog: List[Drug],
    graph: DrugGraphClient,
    age: Optional[float] = None,
    pregnant: bool = False,
    include_ingredient_nodes: bool = True,
    dur_only_subgraph: bool = False,
) -> Dict[str, Any]:
    graph_id = str(uuid.uuid4())
    errors: List[str] = []
    warnings: List[str] = []
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    pair_assessments: List[Dict[str, Any]] = []
    mapping_out: List[Dict[str, Any]] = []

    total_inputs = len([x for x in (drug_ids or []) if str(x).strip()]) + len(
        [x for x in (drug_names or []) if str(x).strip()]
    )
    if total_inputs > MAX_DRUGS:
        return {
            "graph_id": graph_id,
            "ok": False,
            "backend": graph.backend.value,
            "nodes": [],
            "edges": [],
            "mapping": [],
            "pair_assessments": [],
            "data_provenance": provenance_to_dict(DataProvenance(safety=graph.backend)) or {},
            "warnings": [],
            "errors": [f"입력 약물 수는 최대 {MAX_DRUGS}개까지 허용됩니다."],
            "disclaimer": DISCLAIMER,
        }

    if graph.backend == DataBackend.DISABLED:
        errors.append("USE_DE_GRAPH=false — Neo4j 그래프 비활성")
    elif graph.backend == DataBackend.UNAVAILABLE:
        errors.append("Neo4j 연결 불가")

    mapped, map_warnings = normalize_drug_inputs(
        drug_ids, drug_names, catalog=catalog, graph=graph
    )
    warnings.extend(map_warnings)

    for m in mapped:
        mapping_out.append(
            {
                "input": {"kind": m.input_kind, "value": m.input_value},
                "item_seq": m.item_seq,
                "label": m.label,
                "mapping_status": m.mapping_status,
                "graph_status": m.graph_status,
                "candidates_count": len(m.candidates),
            }
        )

    resolved_seqs = [m.item_seq for m in mapped if m.item_seq and m.mapping_status == "RESOLVED"]

    node_index: Dict[str, Dict[str, Any]] = {}

    def ensure_drug_node(m: MappedDrug) -> str:
        if m.item_seq and m.mapping_status == "RESOLVED":
            nid = _drug_node_id(m.item_seq)
            if nid not in node_index:
                node_index[nid] = {
                    "id": nid,
                    "type": "Drug",
                    "label": m.label or m.item_seq,
                    "item_seq": m.item_seq,
                    "graph_status": m.graph_status,
                    "mapping_status": m.mapping_status,
                    "highlight": False,
                }
            return nid
        nid = _unmapped_node_id(m.input_kind, m.input_value)
        if nid not in node_index:
            node_index[nid] = {
                "id": nid,
                "type": "Drug",
                "label": m.label or m.input_value,
                "item_seq": None,
                "graph_status": m.graph_status,
                "mapping_status": m.mapping_status,
                "highlight": m.mapping_status in ("AMBIGUOUS", "NOT_MAPPED"),
            }
        return nid

    for m in mapped:
        ensure_drug_node(m)

    edge_keys: Set[str] = set()
    interaction_edge_keys: Set[str] = set()

    def add_edge(edge: Dict[str, Any]) -> None:
        eid = edge.get("id") or _edge_id(edge["source"], edge["target"], edge.get("type", ""))
        edge["id"] = eid
        if eid in edge_keys:
            return
        edge_keys.add(eid)
        edges.append(edge)

    backend_ok = graph.backend not in (DataBackend.DISABLED, DataBackend.UNAVAILABLE)

    def ensure_ingredient_node(code: str, label: Optional[str] = None) -> str:
        iid = _ingr_node_id(code)
        if iid not in node_index:
            node_index[iid] = {
                "id": iid,
                "type": "Ingredient",
                "label": label or code,
                "code": code,
                "graph_status": "CHECKABLE",
                "mapping_status": "RESOLVED",
                "highlight": False,
            }
        return iid

    pairs: List[Tuple[str, str]] = []
    for i, a in enumerate(resolved_seqs):
        for b in resolved_seqs[i + 1 :]:
            pairs.append(_pair_key(a, b))

    flagged_ingredient_codes: Set[str] = set()

    for a, b in pairs:
        findings_a = graph.check_interactions([a], b)
        findings_b = graph.check_interactions([b], a)
        merged_findings = _merge_bidirectional_findings(findings_a + findings_b)
        drug_a, drug_b = _pair_key(a, b)

        for f in merged_findings:
            if f.get("type") != "contraindication":
                continue
            for code in (f.get("current_code"), f.get("candidate_code")):
                if code:
                    flagged_ingredient_codes.add(str(code))

        for current, candidate in ((a, b), (b, a)):
            assess_out = graph.assess(
                [current],
                candidate,
                age=age,
                pregnant=pregnant,
                with_graph=False,
            )
            assess_findings = (assess_out.payload or {}).get("findings") or []
            pair_assessments.append(
                {
                    "pair": _pair_key(a, b),
                    "evaluation_direction": {
                        "current_item_seqs": [current],
                        "candidate_item_seq": candidate,
                    },
                    "verdict": (
                        str((assess_out.payload or {}).get("verdict") or "UNKNOWN").upper()
                        if assess_out.ok
                        else "UNAVAILABLE"
                    ),
                    "findings": assess_findings,
                    "notes": (assess_out.payload or {}).get("notes") or [],
                    "backend": assess_out.backend.value,
                    "ok": assess_out.ok,
                    "assessment_status": (
                        "completed" if assess_out.ok else "unavailable"
                    ),
                    "patient_context": {"age": age, "pregnant": pregnant},
                }
            )

        for f in merged_findings:
            ftype = str(f.get("type") or "")
            if ftype not in INTERACTION_FINDING_TYPES:
                continue
            level = f.get("level") or "caution"
            src_d = _drug_node_id(drug_a)
            tgt_d = _drug_node_id(drug_b)
            ca, cb = _canonical_ingredient_codes(
                f.get("current_code"), f.get("candidate_code")
            )
            ik = _edge_id("interaction", drug_a, drug_b, ftype, ca, cb)
            if ik in interaction_edge_keys:
                continue
            interaction_edge_keys.add(ik)
            if src_d in node_index and tgt_d in node_index:
                node_index[src_d]["highlight"] = True
                node_index[tgt_d]["highlight"] = True
            evidence = _edge_evidence_fields(f)
            add_edge(
                {
                    "id": ik,
                    "source": src_d,
                    "target": tgt_d,
                    "kind": "interaction",
                    "type": ftype,
                    "directed": False,
                    "level": level,
                    "verdict": _verdict_from_level(level, ftype),
                    "ingredient_pair": [ca, cb] if ca or cb else [],
                    "assessment_status": "interaction_found",
                    **evidence,
                }
            )

        if include_ingredient_nodes:
            for f in merged_findings:
                if f.get("type") != "contraindication":
                    continue
                cc, kc = _canonical_ingredient_codes(
                    f.get("current_code"), f.get("candidate_code")
                )
                if cc and kc:
                    ia = ensure_ingredient_node(cc)
                    ib = ensure_ingredient_node(kc)
                    evidence = _edge_evidence_fields(f)
                    add_edge(
                        {
                            "id": _edge_id("CONTRAINDICATED_WITH", cc, kc),
                            "source": ia,
                            "target": ib,
                            "kind": "structural",
                            "type": "CONTRAINDICATED_WITH",
                            "directed": False,
                            "level": "ban",
                            "verdict": "BAN",
                            "ingredient_pair": [cc, kc],
                            "assessment_status": "interaction_found",
                            **evidence,
                        }
                    )

    if backend_ok and include_ingredient_nodes:
        for seq in resolved_seqs:
            st = graph.check_status(seq)
            did = _drug_node_id(seq)
            for ing in st.get("ingredients") or []:
                if not ing or not ing.get("code"):
                    continue
                code = str(ing["code"])
                if dur_only_subgraph:
                    in_dur = bool(ing.get("in_dur"))
                    if not in_dur and code not in flagged_ingredient_codes:
                        continue
                iid = ensure_ingredient_node(code, ing.get("name_kor") or code)
                if code in flagged_ingredient_codes and iid in node_index:
                    node_index[iid]["highlight"] = True
                add_edge(
                    {
                        "id": _edge_id(did, iid, "CONTAINS"),
                        "source": did,
                        "target": iid,
                        "kind": "structural",
                        "type": "CONTAINS",
                        "directed": True,
                        "level": None,
                        "verdict": None,
                        "message": "",
                        "ingredient_pair": None,
                        "evidence_source": None,
                        "source_url": None,
                    }
                )

    nodes = list(node_index.values())
    provenance = DataProvenance(safety=graph.backend)
    if errors:
        provenance.note("interaction_graph_errors=" + ";".join(errors[:3]))

    ok = not errors and (bool(resolved_seqs) or bool(mapped))
    if not resolved_seqs and mapped and not errors:
        warnings.append("매핑된 item_seq 없음 — 노드만 표시됩니다.")
    if resolved_seqs and not interaction_edge_keys and backend_ok:
        warnings.append(
            "약물 쌍에서 check_interactions findings가 없습니다. "
            "이는 복용 안전(OK)을 의미하지 않습니다."
        )

    return {
        "graph_id": graph_id,
        "ok": ok,
        "backend": graph.backend.value,
        "nodes": nodes,
        "edges": edges,
        "mapping": mapping_out,
        "pair_assessments": pair_assessments,
        "data_provenance": provenance_to_dict(provenance) or {},
        "warnings": warnings,
        "errors": [_sanitize_error(e) for e in errors],
        "disclaimer": DISCLAIMER,
    }
