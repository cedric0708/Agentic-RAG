from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from agents.integrations.types import DataBackend
from agents.models import Drug, UserInput


def normalize_key(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


def item_seq_from_drug_id(drug_id: str) -> str:
    """drugs.json drug_id ↔ Neo4j item_seq (품목기준코드 9자리)."""
    return (drug_id or "").strip()


def build_symptom_queries(user_input: UserInput, user_text: str = "") -> List[str]:
    parts = [s.strip() for s in user_input.symptoms if s.strip()]
    if user_text.strip():
        parts.append(user_text.strip())
    seen: set[str] = set()
    out: List[str] = []
    for p in parts:
        key = normalize_key(p)
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out or ["일반의약품"]


def lookup_drug_by_id(drugs: List[Drug], item_seq: str) -> Optional[Drug]:
    key = item_seq_from_drug_id(item_seq)
    for drug in drugs:
        if drug.drug_id == key:
            return drug
    return None


def lookup_drug_by_name(drugs: List[Drug], name: str) -> Optional[Drug]:
    norm = normalize_key(name)
    if not norm:
        return None
    for drug in drugs:
        if normalize_key(drug.name_ko) == norm:
            return drug
        if drug.name_en and normalize_key(drug.name_en) == norm:
            return drug
    for drug in drugs:
        if norm in normalize_key(drug.name_ko):
            return drug
    return None


def drug_from_de_row(row: dict, catalog: List[Drug]) -> Tuple[Drug, str]:
    """
    DE search_drugs 행 → Drug.
    catalog에 있으면 catalog Drug 사용, 없으면 최소 Drug 스텁(legacy 보조).
    반환: (drug, match_kind)  match_kind: catalog | stub
    """
    item_seq = str(row.get("item_seq") or "").strip()
    name = str(row.get("name") or "").strip() or item_seq or "unknown"

    if item_seq:
        found = lookup_drug_by_id(catalog, item_seq)
        if found:
            return found, "catalog"

    found = lookup_drug_by_name(catalog, name)
    if found:
        return found, "catalog"

    text = str(row.get("text") or "")[:500]
    stub = Drug(
        drug_id=item_seq or normalize_key(name),
        name_ko=name,
        name_en=None,
        ingredient=[],
        indications=text or "DE 검색 문서(로컬 catalog 미매칭)",
        dosage="",
        warnings="",
        category=None,
        updated_date=None,
        combination_contraindication=[],
        parent_text=text,
        child_chunks=[],
    )
    return stub, "stub"


def resolve_current_item_seqs(
    user_input: UserInput,
    drugs: List[Drug],
    *,
    find_drug_fn=None,
) -> Tuple[List[str], List[str]]:
    """
    복용 중 약 → item_seq 목록.
    find_drug_fn: graph find_drug(query) -> list[dict] (선택)
    """
    seqs: List[str] = []
    notes: List[str] = []
    seen: set[str] = set()

    def add(seq: str, note: str = "") -> None:
        seq = (seq or "").strip()
        if not seq or seq in seen:
            return
        seen.add(seq)
        seqs.append(seq)
        if note:
            notes.append(note)

    for drug_id in user_input.current_drug_ids:
        add(item_seq_from_drug_id(drug_id), f"복용약 ID→item_seq: {drug_id}")

    for name in user_input.current_drug_names:
        name = (name or "").strip()
        if not name:
            continue
        local = lookup_drug_by_name(drugs, name)
        if local:
            add(local.drug_id, f"복용약 이름(catalog): {name}")
            continue
        if find_drug_fn:
            try:
                hits = find_drug_fn(name) or []
            except Exception:
                hits = []
            if hits:
                add(str(hits[0].get("item_seq") or ""), f"복용약 이름(Neo4j find_drug): {name}")
                continue
        notes.append(f"복용약 '{name}' item_seq 미확인 — Neo4j assess가 불완전할 수 있음")

    return seqs, notes


def user_is_pregnant(user_input: UserInput) -> bool:
    for c in user_input.conditions:
        norm = normalize_key(c)
        if any(k in norm for k in ("임신", "임산", "pregnant")):
            return True
    return False


def format_backend_label(backend: DataBackend) -> str:
    return backend.value
