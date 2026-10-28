"""Multi-Agent Reasoning Trace — 구조화된 Decision Trace (CoT 아님)."""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from agents.agent_trace import AgentStep

if TYPE_CHECKING:
    from agents.integrations.types import DataBackend, DataProvenance

_logger = logging.getLogger(__name__)

# 비밀정보 마스킹 패턴
_SECRET_PATTERNS = (
    re.compile(r"(?i)(password|passwd|api[_-]?key|secret|token)\s*[:=]\s*\S+"),
    re.compile(r"neo4j(\+s)?://[^:\s]+:[^@\s]+@"),
    re.compile(r"KakaoAK\s+\S+"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, limit: int = 240) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def sanitize_trace_value(value: Any) -> Any:
    """Trace JSON 직렬화 전 비밀정보 제거."""
    if isinstance(value, str):
        out = value
        for pat in _SECRET_PATTERNS:
            out = pat.sub("[REDACTED]", out)
        return out
    if isinstance(value, dict):
        return {k: sanitize_trace_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_trace_value(v) for v in value]
    return value


@dataclass
class ReasoningTraceStep:
    step_id: str
    agent: str
    status: str  # success | skipped | failed | warning
    summary: str = ""
    input_summary: Dict[str, Any] = field(default_factory=dict)
    decision: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    started_at: str = ""
    duration_ms: int = 0
    warnings: List[str] = field(default_factory=list)
    order: int = 0

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        return sanitize_trace_value(raw)


@dataclass
class ReasoningTrace:
    """파이프라인 1회 실행의 전체 Trace (WEB 타임라인용)."""

    trace_id: str
    pipeline_status: str  # completed | emergency_stop | partial
    started_at: str
    completed_at: str = ""
    duration_ms: int = 0
    steps: List[ReasoningTraceStep] = field(default_factory=list)
    data_provenance: Dict[str, Any] = field(default_factory=dict)
    disclaimer: str = (
        "본 Trace는 시스템이 실제 수행한 검색·판정·검증 단계를 기록합니다. "
        "LLM 내부 추론이 아니며, PostgreSQL hit·Neo4j OK가 추천/복용 안전을 보장하지 않습니다."
    )

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_trace_value(
            {
                "trace_id": self.trace_id,
                "pipeline_status": self.pipeline_status,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "duration_ms": self.duration_ms,
                "steps": [s.to_dict() for s in self.steps],
                "data_provenance": self.data_provenance,
                "disclaimer": self.disclaimer,
            }
        )


class ReasoningTraceCollector:
    """파이프라인 실행 중 단계별 Trace 수집."""

    def __init__(self) -> None:
        self.trace_id = str(uuid.uuid4())
        self.started_at = _utc_now_iso()
        self._t0 = time.perf_counter()
        self.steps: List[ReasoningTraceStep] = []
        self._order = 0

    def add_step(
        self,
        *,
        step_id: str,
        agent: str,
        status: str,
        summary: str,
        input_summary: Optional[Dict[str, Any]] = None,
        decision: Optional[Dict[str, Any]] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        data_sources: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        duration_ms: int = 0,
        started_at: Optional[str] = None,
    ) -> ReasoningTraceStep:
        self._order += 1
        step = ReasoningTraceStep(
            step_id=step_id,
            agent=agent,
            status=status,
            summary=summary,
            input_summary=input_summary or {},
            decision=decision or {},
            evidence=evidence or [],
            data_sources=data_sources or [],
            started_at=started_at or _utc_now_iso(),
            duration_ms=max(0, duration_ms),
            warnings=warnings or [],
            order=self._order,
        )
        self.steps.append(step)
        return step

    def track(
        self,
        step_id: str,
        agent: str,
    ) -> "_StepTimer":
        return _StepTimer(self, step_id, agent)

    def add_skipped_pipeline_tail(self, from_step: str) -> None:
        tail = [
            ("retrieval_agent", "Retrieval Agent"),
            ("reranker", "Reranker"),
            ("recommendation_agent", "Recommendation Agent"),
            ("validator", "Validator"),
        ]
        skip = False
        for sid, agent in tail:
            if sid == from_step:
                skip = True
            if skip:
                self.add_step(
                    step_id=sid,
                    agent=agent,
                    status="skipped",
                    summary="응급 분기로 인해 이 단계는 실행되지 않았습니다.",
                    warnings=["pipeline_short_circuit=emergency"],
                )

    def finalize(
        self,
        *,
        pipeline_status: str,
        data_provenance: Optional[Dict[str, Any]] = None,
    ) -> ReasoningTrace:
        elapsed = int((time.perf_counter() - self._t0) * 1000)
        return ReasoningTrace(
            trace_id=self.trace_id,
            pipeline_status=pipeline_status,
            started_at=self.started_at,
            completed_at=_utc_now_iso(),
            duration_ms=elapsed,
            steps=list(self.steps),
            data_provenance=data_provenance or {},
        )

    def to_agent_steps(self) -> List[AgentStep]:
        """기존 format_agent_trace / debug_trace 호환."""
        out: List[AgentStep] = []
        for s in self.steps:
            messages = [s.summary] if s.summary else []
            for w in s.warnings[:4]:
                messages.append(f"⚠ {w}")
            if s.decision:
                for k, v in list(s.decision.items())[:5]:
                    messages.append(f"{k}: {v}")
            out.append(AgentStep(s.agent, f"{s.status} ({s.duration_ms}ms)", messages))
        return out


class _StepTimer:
    def __init__(self, collector: ReasoningTraceCollector, step_id: str, agent: str) -> None:
        self.collector = collector
        self.step_id = step_id
        self.agent = agent
        self.started_at = _utc_now_iso()
        self._t0 = time.perf_counter()
        self.input_summary: Dict[str, Any] = {}
        self.decision: Dict[str, Any] = {}
        self.evidence: List[Dict[str, Any]] = []
        self.data_sources: List[str] = []
        self.warnings: List[str] = []
        self.summary = ""
        self.status = "success"

    def complete(self) -> ReasoningTraceStep:
        ms = int((time.perf_counter() - self._t0) * 1000)
        return self.collector.add_step(
            step_id=self.step_id,
            agent=self.agent,
            status=self.status,
            summary=self.summary,
            input_summary=self.input_summary,
            decision=self.decision,
            evidence=self.evidence,
            data_sources=self.data_sources,
            warnings=self.warnings,
            duration_ms=ms,
            started_at=self.started_at,
        )


def user_input_summary(user_input) -> Dict[str, Any]:
    return {
        "symptoms": list(user_input.symptoms or []),
        "current_drug_ids": list(user_input.current_drug_ids or []),
        "current_drug_names": list(user_input.current_drug_names or []),
        "allergies": list(user_input.allergies or []),
        "conditions": list(user_input.conditions or []),
        "age": user_input.age,
    }


def retrieval_evidence_from_results(items, *, max_snippet: int = 120) -> List[Dict[str, Any]]:
    ev: List[Dict[str, Any]] = []
    for item in items:
        r = item if hasattr(item, "drug") else item
        drug = r.drug if hasattr(r, "drug") else r
        score = getattr(r, "score", None)
        de_score = getattr(r, "de_score", None)
        catalog_match = getattr(r, "catalog_match", "catalog")
        src = getattr(r, "retrieval_source", "legacy_json")
        entry: Dict[str, Any] = {
            "drug_id": drug.drug_id,
            "drug_name": drug.name_ko,
            "retrieval_score": round(float(score), 4) if score is not None else None,
            "de_score": round(float(de_score), 4) if de_score is not None else None,
            "catalog_match": catalog_match,
            "retrieval_source": src,
        }
        if catalog_match == "stub":
            entry["note"] = "로컬 catalog 미매칭 — 상세·성분·용법 미확인"
        snippet = getattr(r, "de_document_snippet", None)
        if snippet:
            entry["document_snippet"] = _truncate(snippet, max_snippet)
        ev.append(entry)
    return ev


def reranker_evidence_from_results(reranked) -> List[Dict[str, Any]]:
    ev: List[Dict[str, Any]] = []
    for item in reranked:
        verdict = None
        safety_backend = None
        for reason in item.rerank_reasons:
            if reason.startswith("neo4j_verdict="):
                verdict = reason.split("=", 1)[1].strip()
            if reason.startswith("safety_backend="):
                safety_backend = reason.split("=", 1)[1].strip()
        ev.append(
            {
                "drug_id": item.retrieval.drug.drug_id,
                "drug_name": item.retrieval.drug.name_ko,
                "final_score": round(item.final_score, 3),
                "risk_level": item.risk_level,
                "blocked": item.blocked,
                "needs_extra_verification": item.needs_extra_verification,
                "neo4j_verdict": verdict,
                "safety_backend": safety_backend,
                "catalog_match": item.retrieval.catalog_match,
            }
        )
    return ev


def recommendation_decision_trace(decision) -> Dict[str, Any]:
    recommended = [
        {
            "drug_id": x.retrieval.drug.drug_id,
            "drug_name": x.retrieval.drug.name_ko,
            "final_score": round(x.final_score, 3),
            "risk_level": x.risk_level,
        }
        for x in decision.recommended
    ]
    rejected = [
        {
            "drug_id": x.retrieval.drug.drug_id,
            "drug_name": x.retrieval.drug.name_ko,
            "blocked": x.blocked,
            "needs_extra_verification": x.needs_extra_verification,
            "risk_level": x.risk_level,
            "reason_hints": [r for r in x.rerank_reasons if "blocked" in r or "stub" in r or "verdict" in r][:3],
        }
        for x in decision.rejected
    ]
    return {
        "recommended_count": len(recommended),
        "rejected_count": len(rejected),
        "recommended": recommended,
        "rejected": rejected,
        "messages": list(decision.messages),
    }


def compute_retrieval_data_sources(
    retrieved,
    retrieval_backend: "DataBackend",
    retrieval_notes: List[str],
) -> List[str]:
    """실제 hit의 retrieval_source와 notes 기준 (HYBRID라도 PostgreSQL 단정 금지)."""
    from agents.integrations.types import DataBackend

    hit_sources: List[str] = []
    seen: set[str] = set()
    for r in retrieved:
        src = (getattr(r, "retrieval_source", None) or DataBackend.LEGACY_JSON.value).strip().lower()
        if src not in seen:
            seen.add(src)
            hit_sources.append(src)

    notes_l = " ".join(retrieval_notes).lower()

    if not retrieved:
        if retrieval_backend == DataBackend.UNAVAILABLE:
            if "fallback" in notes_l or "legacy" in notes_l:
                return [DataBackend.LEGACY_JSON.value]
            return [DataBackend.UNAVAILABLE.value]
        return [retrieval_backend.value]

    sources = list(hit_sources)
    if retrieval_backend == DataBackend.HYBRID:
        if DataBackend.LEGACY_JSON.value not in sources and any(
            "legacy" in n.lower() or "보충" in n for n in retrieval_notes
        ):
            sources.append(DataBackend.LEGACY_JSON.value)
    return sources


def collect_safety_backends(reranked) -> List[str]:
    backends: List[str] = []
    for item in reranked:
        for reason in item.rerank_reasons:
            if reason.startswith("safety_backend="):
                raw = reason.split("=", 1)[1].strip()
                if raw not in backends:
                    backends.append(raw)
    return backends


def apply_safety_provenance(provenance: "DataProvenance", reranked) -> None:
    from agents.integrations.types import DataBackend

    backends = collect_safety_backends(reranked)
    provenance.safety_backends = list(backends)
    if not backends:
        return
    if len(backends) == 1:
        try:
            provenance.safety = DataBackend(backends[0])
        except ValueError:
            provenance.note(f"unknown safety_backend={backends[0]}")
        return
    provenance.safety = DataBackend.HYBRID
    provenance.note(f"safety_mixed={','.join(backends)}")


def legacy_messages_query(query_analysis) -> tuple[str, List[str]]:
    return "입력 분석 완료", list(query_analysis.messages)


def legacy_messages_clarification(clarification) -> tuple[str, List[str]]:
    if clarification.needs_clarification:
        status = "추가 확인 필요"
        messages = [f"Q. {q}" for q in clarification.questions]
        messages.extend(f"Reason: {r}" for r in clarification.reasons)
        return status, messages
    return "추가 질문 불필요", ["추천에 필요한 기본 정보가 충분합니다."]


@dataclass
class PipelineTraceBridge:
    """구조화 Trace와 legacy trace_steps/on_step 이중 기록 (오류 격리)."""

    enable_structured: bool
    trace_steps: List[AgentStep]
    on_step: Optional[Callable[[AgentStep], None]]
    collector: Optional[ReasoningTraceCollector] = None
    trace_errors: List[str] = field(default_factory=list)

    @classmethod
    def create(cls, config, trace_steps: List[AgentStep], on_step) -> "PipelineTraceBridge":
        enabled = bool(getattr(config, "enable_reasoning_trace", True))
        return cls(
            enable_structured=enabled,
            trace_steps=trace_steps,
            on_step=on_step,
            collector=ReasoningTraceCollector() if enabled else None,
        )

    def _record_trace_error(self, where: str, exc: BaseException) -> None:
        self.trace_errors.append(f"{where}:{type(exc).__name__}")
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug("reasoning_trace isolated error at %s (%s)", where, type(exc).__name__)

    def emit_legacy(self, agent: str, status_line: str, messages: List[str]) -> None:
        step = AgentStep(agent, status_line, list(messages))
        self.trace_steps.append(step)
        if self.on_step:
            try:
                self.on_step(step)
            except Exception as exc:
                self._record_trace_error("on_step", exc)

    def emit_step(
        self,
        *,
        step_id: str,
        agent: str,
        legacy_status: str,
        legacy_messages: List[str],
        populate: Optional[Callable[[_StepTimer], None]] = None,
    ) -> None:
        if self.enable_structured and self.collector is not None:
            try:
                timer = self.collector.track(step_id, agent)
                if populate:
                    populate(timer)
                rstep = timer.complete()
                status_with_ms = f"{legacy_status} ({rstep.duration_ms}ms)"
                emit_dual(
                    self.collector,
                    self.trace_steps,
                    self.on_step,
                    rstep,
                    legacy_messages=legacy_messages,
                    legacy_status=status_with_ms,
                )
                return
            except Exception as exc:
                self._record_trace_error(f"emit_step({step_id})", exc)
        self.emit_legacy(agent, legacy_status, legacy_messages)

    def add_skipped_pipeline_tail(self, from_step: str) -> None:
        if not self.enable_structured or self.collector is None:
            return
        try:
            self.collector.add_skipped_pipeline_tail(from_step)
        except Exception as exc:
            self._record_trace_error("add_skipped_pipeline_tail", exc)

    def finalize(self, *, pipeline_status: str, data_provenance: Dict[str, Any]) -> Optional[dict]:
        if not self.enable_structured or self.collector is None:
            return None
        try:
            trace = self.collector.finalize(
                pipeline_status=pipeline_status,
                data_provenance=data_provenance,
            )
            return trace.to_dict()
        except Exception as exc:
            self._record_trace_error("finalize", exc)
            return None


def emit_dual(
    collector: Optional[ReasoningTraceCollector],
    legacy_steps: List[AgentStep],
    on_step: Optional[Callable[[AgentStep], None]],
    step: ReasoningTraceStep,
    *,
    legacy_messages: Optional[List[str]] = None,
    legacy_status: Optional[str] = None,
) -> None:
    """구조화 step 완료 후 legacy AgentStep 동기화 (debug_trace 메시지 우선)."""
    if legacy_messages is not None:
        messages = list(legacy_messages)
    else:
        messages = [step.summary] if step.summary else []
        for w in step.warnings[:3]:
            messages.append(f"⚠ {w}")
    status = legacy_status or f"{step.status} ({step.duration_ms}ms)"
    legacy = AgentStep(step.agent, status, messages)
    legacy_steps.append(legacy)
    if on_step:
        try:
            on_step(legacy)
        except Exception:
            if _logger.isEnabledFor(logging.DEBUG):
                _logger.debug("on_step callback failed", exc_info=True)
