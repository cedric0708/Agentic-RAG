#!/usr/bin/env python3
"""실 Neo4j Drug Interaction Graph E2E (읽기 전용). 사용자 로컬 실행용."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)


def _configure_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _println(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    _configure_utf8()
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    os.environ.setdefault("DE_MOCK", "false")
    os.environ.setdefault("USE_DE_GRAPH", "true")

    from agents.config import load_config
    from agents.integrations.factory import create_graph_client
    from agents.integrations.graph_client import LiveGraphClient
    from agents.integrations.types import DataBackend
    from agents.interaction_graph import build_interaction_graph
    from agents.loader import load_drugs

    cfg = load_config()
    graph = create_graph_client(cfg)
    if graph.backend != DataBackend.NEO4J:
        _println(f"[FAIL] Neo4j live 연결 불가 backend={graph.backend.value}")
        return 2

    data_dir = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
    catalog = load_drugs(data_dir / "drugs.json")

    try:
        payload = build_interaction_graph(
            drug_ids=["200402374", "200000913"],
            drug_names=[],
            catalog=catalog,
            graph=graph,
            include_ingredient_nodes=True,
        )
    finally:
        if isinstance(graph, LiveGraphClient):
            graph.close()

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if "password" in text.lower() or "neo4j://" in text.lower():
        _println("[FAIL] 민감정보가 JSON에 포함된 것으로 보입니다.")
        return 3

    _println("[OK] graph_id=" + payload.get("graph_id", ""))
    inter = [e for e in payload.get("edges", []) if e.get("kind") == "interaction"]
    _println(f"[INFO] nodes={len(payload.get('nodes', []))} interaction_edges={len(inter)}")
    if inter:
        _println(f"[INFO] first interaction verdict={inter[0].get('verdict')}")
    else:
        _println("[WARN] interaction edge 없음 — OK로 단정하지 마세요.")

    try:
        json.dumps(payload)
    except Exception as exc:
        _println(f"[FAIL] JSON 직렬화 실패: {exc}")
        return 4

    _println("[DONE] manual_interaction_graph_e2e 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
