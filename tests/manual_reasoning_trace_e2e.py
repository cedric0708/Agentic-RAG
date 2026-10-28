# -*- coding: utf-8 -*-
"""
Multi-Agent Reasoning Trace — 실 DB E2E 수동 검증 스크립트.

- PostgreSQL/pgvector + Neo4j Aura (Live 클라이언트) 읽기 전용 probe
- Mock(DE_MOCK) 성공을 실 DB 성공으로 보고하지 않음
- credential / URI / host / IP 출력 없음

실행 (Windows PowerShell, 프로젝트 루트):
    .venv\\Scripts\\python.exe tests\\manual_reasoning_trace_e2e.py

DB 연결·추천 품질은 별도 축으로 보고합니다.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# UTF-8 콘솔 (Windows)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from agents.config import AppConfig, load_config
from agents.enhanced_pipeline import run_enhanced_pipeline_core
from agents.integrations.context import build_de_context
from agents.integrations.factory import create_graph_client, create_vector_client
from agents.integrations.types import DataBackend, provenance_to_dict
from agents.integrations.vector_client import (
    LiveVectorClient,
    MockVectorClient,
    probe_postgresql_liveness,
)
from agents.loader import load_drugs, load_symptoms
from agents.models import UserInput
from agents.reasoning_trace import sanitize_trace_value

# --- 민감정보 마스킹 (로그용) ---
_REDACT = (
    re.compile(r"(?i)(password|passwd|api[_-]?key|secret|token)\s*[:=]\s*\S+"),
    re.compile(r"neo4j(\+s)?://[^@\s]+@[^\s]+"),
    re.compile(r"@[0-9]{1,3}(?:\.[0-9]{1,3}){3}"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
)


def _println(text: str = "") -> None:
    """Windows 콘솔 인코딩 오류 시에도 출력이 끊기지 않도록 flush."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
        print(safe, flush=True)


def _safe_text(text: str, limit: int = 400) -> str:
    out = str(text or "")
    for pat in _REDACT:
        out = pat.sub("[REDACTED]", out)
    out = sanitize_trace_value(out) if isinstance(out, str) else str(out)
    if len(out) > limit:
        return out[: limit - 3] + "..."
    return out


def load_project_env() -> List[str]:
    """프로젝트 관례: .env 및 config/.env (파일 내용은 출력하지 않음)."""
    loaded: List[str] = []
    candidates = [
        ROOT / ".env",
        ROOT / "config" / ".env",
        ROOT / "de" / "config" / ".env",
        ROOT / "database" / "config" / ".env",
    ]
    for path in candidates:
        if path.is_file():
            load_dotenv(path, override=False)
            loaded.append(str(path.relative_to(ROOT)))
    return loaded


class Verdict:
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class ScenarioResult:
    name: str
    status: str
    lines: List[str]
    infra_only: bool = False


@dataclass
class InfraProbe:
    """인프라 probe: PG connection은 임베딩 없음. 벡터 검색(E5)은 Scenario A에서 검증."""

    postgresql_connection: bool
    neo4j_live: bool
    de_mock_env: bool
    pg_backend: str
    pg_connection_detail: str
    neo4j_backend: str
    neo4j_detail: str
    env_files_loaded: List[str]

    @property
    def postgresql_live(self) -> bool:
        """하위 호환: connection YES (벡터 검색 성공과 동일하지 않음)."""
        return self.postgresql_connection


def probe_infrastructure(cfg: AppConfig) -> InfraProbe:
    """실 DB 연결 여부만 판별 (Mock ≠ Live)."""
    de_mock_env = cfg.de_mock or os.getenv("DE_MOCK", "false").lower() in {"1", "true", "yes"}

    pg_conn = False
    pg_backend = "unknown"
    pg_conn_detail = ""

    if cfg.de_mock:
        pg_backend = DataBackend.MOCK.value
        pg_conn_detail = "DE_MOCK=true — PostgreSQL connection probe 생략"
    elif not cfg.use_de_vector:
        pg_backend = DataBackend.DISABLED.value
        pg_conn_detail = "USE_DE_VECTOR=false"
    else:
        client = create_vector_client(cfg)
        if isinstance(client, MockVectorClient):
            pg_backend = DataBackend.MOCK.value
            pg_conn_detail = "팩토리가 MockVectorClient 반환 (connection probe 해당 없음)"
        elif isinstance(client, LiveVectorClient):
            pg_backend = DataBackend.POSTGRESQL.value
            pg_conn, raw_detail = probe_postgresql_liveness()
            pg_conn_detail = _safe_text(raw_detail, 200)
        else:
            pg_backend = getattr(client, "backend", DataBackend.DISABLED)
            if hasattr(pg_backend, "value"):
                pg_backend = pg_backend.value
            pg_conn_detail = "비-Live 벡터 클라이언트"

    neo_live = False
    neo_backend = "unknown"
    neo4j_detail = ""

    if cfg.de_mock:
        neo_backend = DataBackend.MOCK.value
        neo4j_detail = "DE_MOCK=true — Neo4j Live probe 생략"
    elif not cfg.use_de_graph:
        neo_backend = DataBackend.DISABLED.value
        neo4j_detail = "USE_DE_GRAPH=false"
    else:
        graph = create_graph_client(cfg)
        neo_backend = graph.backend.value
        try:
            hits = graph.find_drug("리피스탄", limit=1)
            if hits and graph.backend == DataBackend.NEO4J:
                neo_live = True
                neo4j_detail = f"find_drug hits={len(hits)} item_seq={hits[0].get('item_seq', '?')}"
            else:
                neo4j_detail = _safe_text(
                    f"backend={graph.backend.value} hits={len(hits or [])}"
                )
        except Exception as exc:
            neo4j_detail = _safe_text(f"{type(exc).__name__}: {exc}")
        finally:
            if hasattr(graph, "close"):
                graph.close()

    return InfraProbe(
        postgresql_connection=pg_conn,
        neo4j_live=neo_live,
        de_mock_env=de_mock_env,
        pg_backend=pg_backend,
        pg_connection_detail=pg_conn_detail,
        neo4j_backend=neo_backend,
        neo4j_detail=neo4j_detail,
        env_files_loaded=[],
    )


def _step_by_id(trace: Optional[dict], step_id: str) -> Optional[dict]:
    if not trace:
        return None
    for s in trace.get("steps") or []:
        if s.get("step_id") == step_id:
            return s
    return None


def _drug_ids(decision) -> Tuple[List[str], List[str]]:
    rec = [x.retrieval.drug.drug_id for x in decision.recommended]
    rej = [x.retrieval.drug.drug_id for x in decision.rejected]
    return rec, rej


def _print_scenario(result: ScenarioResult) -> None:
    _println()
    _println("=" * 72)
    _println(f"시나리오: {result.name}")
    _println(f"결과: {result.status}" + (" [인프라 연결 축]" if result.infra_only else ""))
    _println("-" * 72)
    for line in result.lines:
        _println(line)
    _println("=" * 72)


def resolve_scenario_keys(scenario_arg: str) -> List[str]:
    """--scenario a|b|c|all → 실행할 키 목록."""
    key = (scenario_arg or "all").strip().lower()
    if key == "all":
        return ["a", "b", "c"]
    if key in ("a", "b", "c"):
        return [key]
    raise ValueError(f"unsupported scenario: {scenario_arg!r}")


def execute_scenarios(
    keys: List[str],
    cfg: AppConfig,
    drugs,
    symptoms,
    probe: InfraProbe,
) -> List[ScenarioResult]:
    results: List[ScenarioResult] = []
    runners = {
        "a": run_scenario_a,
        "b": run_scenario_b,
        "c": run_scenario_c,
    }
    labels = {"a": "A. 일반 증상", "b": "B. 병용금기", "c": "C. Trace ON/OFF"}
    for key in keys:
        runner = runners[key]
        _println()
        _println(f">>> 시나리오 {key.upper()} ({labels.get(key, key)}) 실행 중...")
        try:
            results.append(runner(cfg, drugs, symptoms, probe))
        except Exception as exc:
            results.append(
                ScenarioResult(
                    name=labels.get(key, key),
                    status=Verdict.FAIL,
                    lines=[f"시나리오 실행 중 예외: {_safe_text(exc)}"],
                )
            )
    return results


def print_summary(probe: InfraProbe, results: List[ScenarioResult]) -> None:
    _println()
    _println("=" * 72)
    _println("전체 요약")
    _println("=" * 72)
    _println(
        f"PostgreSQL connection: {'YES' if probe.postgresql_connection else 'NO'} | "
        f"Neo4j Live: {'YES' if probe.neo4j_live else 'NO'}"
    )
    n_pass = sum(1 for r in results if r.status == Verdict.PASS)
    n_fail = sum(1 for r in results if r.status == Verdict.FAIL)
    n_skip = sum(1 for r in results if r.status == Verdict.SKIP)
    _println(f"집계: PASS={n_pass} FAIL={n_fail} SKIP={n_skip} (실행={len(results)})")
    if not results:
        _println("ERROR: 실행된 시나리오가 없습니다.")
    for r in results:
        _println(f"  [{r.status}] {r.name}")
    _println()
    _println("해석 가이드:")
    _println("  - SKIP: DB 미연결/Mock — 인프라 미충족, 품질 실패와 구분")
    _println("  - A/B의 PG·Neo4j 항목: 연결·출처·assess 계약")
    _println("  - 추천 약 ID·검색 hit: 임상/매핑 품질 — Live 연결만으로 PASS 아님")
    _println("=" * 72)


def compute_exit_code(results: List[ScenarioResult]) -> int:
    if not results:
        return 1
    if any(r.status == Verdict.FAIL for r in results):
        return 1
    return 0


def run_scenario_a(cfg: AppConfig, drugs, symptoms, probe: InfraProbe) -> ScenarioResult:
    name = "A. 일반 증상 검색 (한글)"
    lines: List[str] = []
    status = Verdict.PASS

    ui = UserInput(symptoms=["머리가 아프고 미열이 있어요"])
    run_cfg = replace(cfg, enable_reasoning_trace=True, use_chroma=False)
    try:
        result = run_enhanced_pipeline_core(
            ui, drugs, symptoms, top_k=3, config=run_cfg, context_text="두통과 미열"
        )
    except Exception as exc:
        return ScenarioResult(name, Verdict.FAIL, [f"파이프라인 예외: {_safe_text(exc)}"])

    prov = result.data_provenance
    prov_dict = provenance_to_dict(prov) or {}
    retrieval_step = _step_by_id(result.reasoning_trace, "retrieval_agent")

    lines.append(f"입력 증상: {ui.symptoms}")
    lines.append(f"Run retrieval: {prov.retrieval.value}")
    lines.append(f"Run safety: {prov.safety.value}")
    if prov.retrieval_hit_sources:
        lines.append(f"retrieval_hit_sources: {prov.retrieval_hit_sources}")
    if prov.safety_backends:
        lines.append(f"safety_backends: {prov.safety_backends}")

    hit_count = 0
    backend_decision = None
    hit_sources_trace: List[str] = []
    evidence_sources: List[str] = []
    if retrieval_step:
        hit_count = retrieval_step.get("decision", {}).get("retrieved_count", 0)
        backend_decision = retrieval_step.get("decision", {}).get("backend")
        hit_sources_trace = retrieval_step.get("decision", {}).get("hit_sources") or []
        hit_sources_trace = hit_sources_trace or retrieval_step.get("data_sources") or []
        for ev in retrieval_step.get("evidence") or []:
            src = ev.get("retrieval_source")
            if src and src not in evidence_sources:
                evidence_sources.append(src)
        lines.append(f"Retrieval step decision.backend: {backend_decision}")
        lines.append(f"Retrieval step data_sources: {retrieval_step.get('data_sources')}")
        lines.append(f"Retrieval evidence retrieval_source(s): {evidence_sources or '(none)'}")
    else:
        lines.append("reasoning_trace retrieval_agent step 없음")
        status = Verdict.FAIL

    rec_ids, rej_ids = _drug_ids(result.decision)
    lines.append(f"Retrieval hit 수(Trace): {hit_count}")
    lines.append(f"추천 {len(rec_ids)}건 ID: {rec_ids}")
    lines.append(f"제외 {len(rej_ids)}건 ID: {rej_ids}")
    lines.append(f"Validator passed: {result.validation.passed}")

    if result.reasoning_trace:
        steps = result.reasoning_trace.get("steps") or []
        lines.append("Trace steps (agent / status):")
        for s in steps:
            lines.append(f"  - {s.get('agent')}: {s.get('status')} | {s.get('summary', '')[:80]}")
        prov_trace = result.reasoning_trace.get("data_provenance") or {}
        match = prov_trace.get("retrieval") == prov_dict.get("retrieval")
        lines.append(f"DataProvenance Trace↔Run retrieval 일치: {match}")
        if not match:
            status = Verdict.FAIL
    else:
        status = Verdict.FAIL
        lines.append("reasoning_trace 없음 (ENABLE_REASONING_TRACE 확인)")

    lines.append(
        f"PostgreSQL connection (probe): "
        f"{'YES' if probe.postgresql_connection else 'NO'} — "
        f"{_safe_text(probe.pg_connection_detail, 120)}"
    )

    live_markers = ("postgresql", "hybrid")
    vector_used = (
        prov.retrieval.value in live_markers
        or backend_decision in live_markers
        or "postgresql" in (hit_sources_trace or evidence_sources)
    )
    vector_pass = vector_used and hit_count > 0 and not (
        DataBackend.MOCK.value in (hit_sources_trace or evidence_sources)
        and "postgresql" not in (hit_sources_trace or evidence_sources)
    )
    if vector_pass:
        lines.append("Vector retrieval (E5 + PostgreSQL): PASS")
    elif probe.postgresql_connection and cfg.use_de_vector and not cfg.de_mock:
        if status == Verdict.PASS:
            status = Verdict.FAIL
        lines.append("Vector retrieval (E5 + PostgreSQL): FAIL — PG 연결은 되나 벡터 검색/출처 미확인")
    else:
        lines.append("Vector retrieval (E5 + PostgreSQL): SKIP — connection 또는 DE vector 비활성")
        if not probe.postgresql_connection and status == Verdict.PASS:
            status = Verdict.SKIP

    if probe.postgresql_connection and vector_used and DataBackend.MOCK.value in (
        hit_sources_trace or evidence_sources
    ):
        if "postgresql" not in (hit_sources_trace or evidence_sources):
            status = Verdict.FAIL
            lines.append("FAIL: Mock 출처만 확인 — DE_MOCK 또는 fallback 확인")

    lines.append("")
    lines.append(
        "[축 구분] 위 PG/Safety backend는 '연결·출처' 검증입니다. "
        "추천 약 ID의 임상적 적합성은 별도 품질 검증이 필요합니다."
    )
    return ScenarioResult(name, status, lines)


def _graph_ban_probe(cfg: AppConfig) -> Tuple[bool, List[str], Optional[str]]:
    """리피스탄(심바스타틴) + 코니트라(이트라코나졸) — Graph assess (파이프라인과 분리)."""
    lines: List[str] = []
    graph = create_graph_client(cfg)
    try:
        if graph.backend != DataBackend.NEO4J:
            return False, [f"Neo4j Live 아님: backend={graph.backend.value}"], None

        lip = graph.find_drug("리피스탄", limit=3)
        kon = graph.find_drug("코니트라", limit=3)
        lines.append(f"find_drug 리피스탄 hits={len(lip or [])}")
        lines.append(f"find_drug 코니트라 hits={len(kon or [])}")
        if not lip or not kon:
            lines.append("Graph에서 품목 item_seq 미발견 — 이름/DB 동기화 확인")
            return False, lines, None

        lip_seq = str(lip[0].get("item_seq") or "").strip()
        kon_seq = str(kon[0].get("item_seq") or "").strip()
        lip_name = lip[0].get("name") or lip_seq
        kon_name = kon[0].get("name") or kon_seq
        lines.append(f"current item_seq: {lip_seq} ({lip_name})")
        lines.append(f"candidate item_seq: {kon_seq} ({kon_name})")

        outcome = graph.assess([lip_seq], kon_seq)
        lines.append(f"assess ok={outcome.ok} backend={outcome.backend.value}")
        if not outcome.ok or not outcome.payload:
            lines.append(f"assess 실패: {_safe_text(outcome.error or 'no payload')}")
            return False, lines, None

        verdict = str(outcome.payload.get("verdict") or "").upper()
        findings = outcome.payload.get("findings") or []
        lines.append(f"verdict: {verdict}")
        lines.append(f"findings count: {len(findings)}")
        for f in findings[:3]:
            lines.append(
                f"  - type={f.get('type')} msg={_safe_text(str(f.get('message') or ''), 120)}"
            )
        if verdict == "BAN":
            lines.append("Graph assess: BAN 확인 (병용금기 축)")
        elif verdict in ("UNKNOWN", "NOT_FOUND"):
            lines.append(
                "Graph assess: BAN 아님 — UNKNOWN/NOT_FOUND는 병용금기 확정으로 해석하지 않음"
            )
        else:
            lines.append(f"Graph assess: 기대와 다른 verdict={verdict} (DB 데이터·쿼리 확인)")

        return verdict == "BAN", lines, verdict
    finally:
        if hasattr(graph, "close"):
            graph.close()


def run_scenario_b(cfg: AppConfig, drugs, symptoms, probe: InfraProbe) -> ScenarioResult:
    name = "B. 병용금기 (리피스탄 + 코니트라)"
    lines: List[str] = []
    status = Verdict.PASS

    lines.append("--- B-1. Graph Adapter 직접 assess (Neo4j 계약) ---")
    if not probe.neo4j_live:
        status = Verdict.SKIP
        lines.append("SKIP(인프라): Neo4j Live probe 실패")
        lines.append(f"  Neo4j probe: {_safe_text(probe.neo4j_detail)}")
    else:
        ban_ok, graph_lines, verdict = _graph_ban_probe(cfg)
        lines.extend(graph_lines)
        if not ban_ok:
            status = Verdict.FAIL if probe.neo4j_live else Verdict.SKIP
            if verdict in ("UNKNOWN", "NOT_FOUND"):
                lines.append(
                    "FAIL(판정): BAN 기대 시나리오에서 UNKNOWN/NOT_FOUND — "
                    "BAN으로 표기하면 안 됨 (현재 verdict는 BAN 아님)"
                )

    lines.append("")
    lines.append("--- B-2. 전체 파이프라인 (검색·추천 품질 축, BAN 강제 아님) ---")
    lines.append(
        "복용약=리피스탄정, 증상=무좀/발톱 — PG 검색이 코니트라를 후보로 올리지 않으면 "
        "Reranker BAN evidence가 없을 수 있음 (정상)."
    )

    ui = UserInput(
        symptoms=["발톱무좀", "무좀"],
        current_drug_names=["리피스탄정"],
    )
    run_cfg = replace(cfg, enable_reasoning_trace=True, use_chroma=False)
    try:
        result = run_enhanced_pipeline_core(
            ui, drugs, symptoms, top_k=5, config=run_cfg, context_text="발톱무좀 치료"
        )
    except Exception as exc:
        lines.append(f"파이프라인 예외: {_safe_text(exc)}")
        return ScenarioResult(name, Verdict.FAIL, lines)

    prov = result.data_provenance
    lines.append(f"Run safety: {prov.safety.value} backends={prov.safety_backends}")
    rerank = _step_by_id(result.reasoning_trace, "reranker")
    safety_step = _step_by_id(result.reasoning_trace, "safety_agent")
    verdicts: List[str] = []
    if rerank:
        for ev in rerank.get("evidence") or []:
            v = ev.get("neo4j_verdict")
            if v:
                verdicts.append(f"{ev.get('drug_name')}: {v} (backend={ev.get('safety_backend')})")
        lines.append("Reranker neo4j_verdict:")
        lines.extend([f"  - {x}" for x in verdicts] or ["  (none)"])
        mislabeled = [
            v
            for v in (ev.get("neo4j_verdict") for ev in (rerank.get("evidence") or []))
            if v == "BAN"
        ]
        unknowns = [
            ev
            for ev in (rerank.get("evidence") or [])
            if ev.get("neo4j_verdict") in ("UNKNOWN", "NOT_FOUND")
        ]
        for ev in unknowns:
            hints = " ".join(str(w) for w in rerank.get("warnings") or [])
            if ev.get("drug_name") and "병용금기" in hints and ev.get("neo4j_verdict") == "UNKNOWN":
                lines.append(
                    f"주의: {ev.get('drug_name')} UNKNOWN인데 병용금기 확정 문구 가능성 — Trace warnings 확인"
                )
    if safety_step:
        lines.append(f"Safety Agent decision: {safety_step.get('decision')}")

    rec_ids, rej_ids = _drug_ids(result.decision)
    lines.append(f"추천 ID: {rec_ids}")
    lines.append(f"제외 ID: {rej_ids}")

    if probe.neo4j_live and status == Verdict.PASS:
        # B-2는 품질 축: BAN trace 없어도 B-1 PASS면 시나리오 PASS 유지
        if not verdicts:
            lines.append(
                "참고: 파이프라인 Reranker에 Neo4j verdict 없음 — 후보 미검색 또는 legacy-only safety"
            )
    lines.append("")
    lines.append(
        "[축 구분] B-1=Neo4j 연결·assess 계약. B-2=검색/매핑/추천 품질(연결 성공 ≠ 품질 PASS)."
    )
    return ScenarioResult(name, status, lines)


def run_scenario_c(cfg: AppConfig, drugs, symptoms, _probe: InfraProbe) -> ScenarioResult:
    name = "C. Trace ON/OFF 비교"
    lines: List[str] = []
    status = Verdict.PASS

    ui = UserInput(symptoms=["소화불량", "속 쓰림"])
    base = replace(cfg, use_chroma=False)
    cfg_on = replace(base, enable_reasoning_trace=True)
    cfg_off = replace(base, enable_reasoning_trace=False)

    try:
        on = run_enhanced_pipeline_core(ui, drugs, symptoms, top_k=3, config=cfg_on)
        off = run_enhanced_pipeline_core(ui, drugs, symptoms, top_k=3, config=cfg_off)
    except Exception as exc:
        return ScenarioResult(name, Verdict.FAIL, [f"파이프라인 예외: {_safe_text(exc)}"])

    on_rec, on_rej = _drug_ids(on.decision)
    off_rec, off_rej = _drug_ids(off.decision)
    lines.append(f"Trace ON  — recommended: {on_rec} rejected: {on_rej}")
    lines.append(f"Trace OFF — recommended: {off_rec} rejected: {off_rej}")
    lines.append(f"Validator ON/OFF: {on.validation.passed} / {off.validation.passed}")

    if on.reasoning_trace is None:
        status = Verdict.FAIL
        lines.append("FAIL: Trace ON인데 reasoning_trace is None")
    else:
        lines.append(
            f"Trace ON trace_id: {on.reasoning_trace.get('trace_id')} "
            f"steps={len(on.reasoning_trace.get('steps') or [])}"
        )
    if off.reasoning_trace is not None:
        status = Verdict.FAIL
        lines.append("FAIL: Trace OFF인데 reasoning_trace가 None이 아님")

    if on_rec != off_rec or on_rej != off_rej:
        status = Verdict.FAIL
        lines.append("FAIL: 추천/제외 ID 불일치")
    elif on.validation.passed != off.validation.passed:
        status = Verdict.FAIL
        lines.append("FAIL: Validator 결과 불일치")
    else:
        lines.append("PASS: Trace flag만 다르고 decision/validation 동일")

    lines.append(
        "참고: PG 벡터 검색 tie-break 등으로 환경에 따라 ID 순서가 달라질 수 있음 — "
        "집합 불일치 시에만 FAIL 처리함."
    )
    return ScenarioResult(name, status, lines)


def print_infra_banner(probe: InfraProbe, env_loaded: List[str]) -> None:
    _println()
    _println("#" * 72)
    _println("# 인프라 연결 probe (추천 품질과 별개)")
    _println("#" * 72)
    if env_loaded:
        _println(f"로드한 env 파일(경로만): {', '.join(env_loaded)}")
    else:
        _println("로드한 env 파일 없음 — OS 환경변수만 사용")
    _println(f"DE_MOCK (effective): {probe.de_mock_env}")
    _println(
        f"PostgreSQL connection: {'YES' if probe.postgresql_connection else 'NO'} "
        f"(backend={probe.pg_backend}) — {_safe_text(probe.pg_connection_detail, 200)}"
    )
    _println(
        "Vector retrieval (E5+PG): Scenario A 파이프라인에서 검증 (probe에서는 E5 로드 없음)"
    )
    _println(
        f"Neo4j Live: {'YES' if probe.neo4j_live else 'NO'} "
        f"(backend={probe.neo4j_backend}) — {_safe_text(probe.neo4j_detail, 200)}"
    )
    if probe.de_mock_env:
        _println("WARN: DE_MOCK=true — Mock 경로는 실 DB 성공으로 보고하지 않습니다.")
    _println("#" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reasoning Trace 실 DB E2E (수동)")
    parser.add_argument(
        "--scenario",
        choices=("a", "b", "c", "all"),
        default="all",
        help="실행할 시나리오",
    )
    parser.add_argument(
        "--keep-de-mock",
        action="store_true",
        help="DE_MOCK 환경값 유지 (기본: E2E는 프로세스 내 DE_MOCK=false 강제)",
    )
    args = parser.parse_args()

    env_loaded = load_project_env()
    if not args.keep_de_mock:
        os.environ["DE_MOCK"] = "false"

    cfg = load_config()
    drugs = load_drugs(cfg.data_dir / "drugs.json")
    symptoms = load_symptoms(cfg.data_dir / "symptoms.json")

    probe = probe_infrastructure(cfg)
    probe.env_files_loaded = env_loaded
    print_infra_banner(probe, env_loaded)

    try:
        scenario_keys = resolve_scenario_keys(args.scenario)
    except ValueError as exc:
        _println(f"ERROR: {exc}")
        return 1

    _println(f"실행 예정 시나리오: {', '.join(k.upper() for k in scenario_keys)}")

    results = execute_scenarios(scenario_keys, cfg, drugs, symptoms, probe)

    for r in results:
        _print_scenario(r)

    print_summary(probe, results)
    return compute_exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
