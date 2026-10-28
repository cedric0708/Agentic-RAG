from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from agents.integrations.types import AssessOutcome, DataBackend


class DrugGraphClient(ABC):
    @property
    @abstractmethod
    def backend(self) -> DataBackend:
        ...

    @abstractmethod
    def assess(
        self,
        current: List[str],
        candidate: str,
        *,
        age: Optional[float] = None,
        pregnant: bool = False,
        with_graph: bool = False,
    ) -> AssessOutcome:
        ...

    @abstractmethod
    def find_drug(self, query: str, limit: int = 5) -> List[dict]:
        ...

    def check_status(self, item_seq: str) -> Dict[str, Any]:
        """Neo4j 제품·성분 상태 (DISABLED/UNAVAILABLE 시 NOT_FOUND 형태)."""
        return {
            "item_seq": item_seq,
            "status": "NOT_FOUND",
            "drug": None,
            "ingredients": [],
        }

    def check_interactions(
        self,
        current: List[str],
        candidate: str,
        *,
        all_duplicates: bool = False,
    ) -> List[dict]:
        return []

    def get_subgraph(
        self,
        current: List[str],
        candidate: str,
        *,
        dur_only: bool = False,
    ) -> Dict[str, Any]:
        return {"nodes": [], "edges": []}


class DisabledGraphClient(DrugGraphClient):
    @property
    def backend(self) -> DataBackend:
        return DataBackend.DISABLED

    def assess(
        self,
        current: List[str],
        candidate: str,
        *,
        age: Optional[float] = None,
        pregnant: bool = False,
        with_graph: bool = False,
    ) -> AssessOutcome:
        return AssessOutcome(ok=False, backend=DataBackend.DISABLED, error="USE_DE_GRAPH=false")

    def find_drug(self, query: str, limit: int = 5) -> List[dict]:
        return []


class LiveGraphClient(DrugGraphClient):
    def __init__(self, *, connection_timeout: float = 8.0) -> None:
        self._graph = None
        self._connect_error: str | None = None
        self._connection_timeout = connection_timeout

    @property
    def backend(self) -> DataBackend:
        if self._connect_error:
            return DataBackend.UNAVAILABLE
        return DataBackend.NEO4J

    def _ensure(self) -> bool:
        if self._graph is not None:
            return True
        if self._connect_error:
            return False
        try:
            import os

            from agents.integrations.de_import import load_de_module
            from neo4j import GraphDatabase

            mod = load_de_module("graph_tools")
            timeout = float(os.getenv("NEO4J_CONNECT_TIMEOUT", str(self._connection_timeout)))
            self._graph = mod.DrugGraph()
            try:
                old = self._graph.driver
                uri = getattr(old, "_uri", None) or os.getenv("NEO4J_URI")
                auth = getattr(old, "_auth", None)
                if uri and auth is not None:
                    old.close()
                    self._graph.driver = GraphDatabase.driver(
                        uri,
                        auth=auth,
                        connection_timeout=timeout,
                    )
            except Exception:
                pass
            return True
        except Exception as exc:
            self._connect_error = str(exc)
            self._graph = None
            return False

    def assess(
        self,
        current: List[str],
        candidate: str,
        *,
        age: Optional[float] = None,
        pregnant: bool = False,
        with_graph: bool = False,
    ) -> AssessOutcome:
        if not self._ensure():
            return AssessOutcome(
                ok=False,
                backend=DataBackend.UNAVAILABLE,
                error=self._connect_error or "Neo4j 연결 실패",
                attempted_neo4j=True,
            )
        try:
            payload = self._graph.assess(
                current,
                candidate,
                age=age,
                pregnant=pregnant,
                with_graph=with_graph,
            )
            return AssessOutcome(
                ok=True,
                payload=payload,
                backend=DataBackend.NEO4J,
                attempted_neo4j=True,
            )
        except Exception as exc:
            return AssessOutcome(
                ok=False,
                backend=DataBackend.UNAVAILABLE,
                error=str(exc),
                attempted_neo4j=True,
            )

    def find_drug(self, query: str, limit: int = 5) -> List[dict]:
        if not self._ensure():
            return []
        try:
            return list(self._graph.find_drug(query, limit=limit) or [])
        except Exception:
            return []

    def check_status(self, item_seq: str) -> Dict[str, Any]:
        if not self._ensure():
            return {
                "item_seq": item_seq,
                "status": "NOT_FOUND",
                "drug": None,
                "ingredients": [],
            }
        try:
            return dict(self._graph.check_status(item_seq))
        except Exception:
            return {
                "item_seq": item_seq,
                "status": "NOT_FOUND",
                "drug": None,
                "ingredients": [],
            }

    def check_interactions(
        self,
        current: List[str],
        candidate: str,
        *,
        all_duplicates: bool = False,
    ) -> List[dict]:
        if not self._ensure() or not current:
            return []
        try:
            return list(
                self._graph.check_interactions(
                    current, candidate, all_duplicates=all_duplicates
                )
                or []
            )
        except Exception:
            return []

    def get_subgraph(
        self,
        current: List[str],
        candidate: str,
        *,
        dur_only: bool = False,
    ) -> Dict[str, Any]:
        if not self._ensure():
            return {"nodes": [], "edges": []}
        try:
            return dict(self._graph.get_subgraph(current, candidate, dur_only=dur_only))
        except Exception:
            return {"nodes": [], "edges": []}

    def close(self) -> None:
        if self._graph is not None:
            try:
                self._graph.close()
            except Exception:
                pass
            self._graph = None


class MockGraphClient(DrugGraphClient):
    """테스트용 assess/find_drug. ban_pairs 또는 candidate별 fixed_verdict 지원."""

    def __init__(
        self,
        *,
        ban_pairs: Optional[List[tuple[str, str]]] = None,
        find_drug_fn: Optional[Callable[[str, int], List[dict]]] = None,
        fixed_verdict: Optional[dict[str, str]] = None,
        status_map: Optional[dict[str, str]] = None,
        fail: bool = False,
        interaction_fn: Optional[
            Callable[[List[str], str], List[dict]]
        ] = None,
    ) -> None:
        self._ban_pairs = set(ban_pairs or [])
        self._find_drug_fn = find_drug_fn
        self._fixed_verdict = {k: v.upper() for k, v in (fixed_verdict or {}).items()}
        self._status_map = status_map or {}
        self._fail = fail
        self._interaction_fn = interaction_fn

    @property
    def backend(self) -> DataBackend:
        return DataBackend.MOCK

    def assess(
        self,
        current: List[str],
        candidate: str,
        *,
        age: Optional[float] = None,
        pregnant: bool = False,
        with_graph: bool = False,
    ) -> AssessOutcome:
        if self._fail:
            return AssessOutcome(
                ok=False,
                backend=DataBackend.UNAVAILABLE,
                error="mock connection failed",
                attempted_neo4j=True,
            )

        if candidate in self._fixed_verdict:
            verdict = self._fixed_verdict[candidate]
            findings: List[dict] = []
            if verdict == "BAN":
                findings.append(
                    {"type": "contraindication", "level": "ban", "message": "Mock 병용금기"}
                )
            payload = {
                "candidate": {"item_seq": candidate},
                "verdict": verdict,
                "findings": findings,
                "notes": [f"Mock fixed verdict={verdict}"],
                "disclaimer": "mock",
            }
            if with_graph:
                payload["graph"] = {"nodes": [], "edges": []}
            return AssessOutcome(
                ok=True, payload=payload, backend=DataBackend.MOCK, attempted_neo4j=True
            )

        verdict = "OK"
        findings = []
        notes: List[str] = []

        for cur in current:
            if (cur, candidate) in self._ban_pairs or (candidate, cur) in self._ban_pairs:
                verdict = "BAN"
                findings.append(
                    {
                        "type": "contraindication",
                        "level": "ban",
                        "message": "Mock 병용금기",
                        "current_seq": cur,
                        "candidate_code": "MOCK",
                        "source": "mock",
                    }
                )

        if pregnant:
            verdict = "BAN" if verdict != "BAN" else verdict
            if verdict == "BAN" and not any(f.get("type") == "contraindication" for f in findings):
                findings.append({"type": "restriction", "level": "ban", "message": "Mock 임부금기"})

        payload = {
            "candidate": {"item_seq": candidate},
            "verdict": verdict,
            "findings": findings,
            "notes": notes or ["Mock Neo4j assess"],
            "disclaimer": "mock",
        }
        if with_graph:
            payload["graph"] = {"nodes": [], "edges": []}

        return AssessOutcome(ok=True, payload=payload, backend=DataBackend.MOCK, attempted_neo4j=True)

    def find_drug(self, query: str, limit: int = 5) -> List[dict]:
        if self._find_drug_fn:
            return self._find_drug_fn(query, limit)
        q = (query or "").replace(" ", "")
        if "활명" in q:
            return [{"item_seq": "195700020", "name": "활명수"}]
        return []

    def check_status(self, item_seq: str) -> Dict[str, Any]:
        st = self._status_map.get(item_seq, "CHECKABLE")
        if st == "NOT_FOUND":
            return {"item_seq": item_seq, "status": "NOT_FOUND", "drug": None, "ingredients": []}
        if st == "UNKNOWN":
            return {
                "item_seq": item_seq,
                "status": "UNKNOWN",
                "drug": {"item_seq": item_seq, "name": f"Drug-{item_seq}"},
                "ingredients": [],
            }
        return {
            "item_seq": item_seq,
            "status": "CHECKABLE",
            "drug": {"item_seq": item_seq, "name": f"Drug-{item_seq}"},
            "ingredients": [{"code": "D000001", "name_kor": "mock-ing", "in_dur": True}],
        }

    def check_interactions(self, current: List[str], candidate: str, *, all_duplicates: bool = False) -> List[dict]:
        if self._interaction_fn:
            return self._interaction_fn(current, candidate)
        if not current:
            return []
        rows: List[dict] = []
        for cur in current:
            if (cur, candidate) in self._ban_pairs or (candidate, cur) in self._ban_pairs:
                rows.append(
                    {
                        "type": "contraindication",
                        "level": "ban",
                        "message": "Mock 병용금기",
                        "current_seq": cur,
                        "candidate_code": "D000002",
                        "current_code": "D000001",
                        "source": "mock-dur",
                        "source_url": "https://example.invalid/mock",
                    }
                )
        return rows

    def get_subgraph(
        self,
        current: List[str],
        candidate: str,
        *,
        dur_only: bool = False,
    ) -> Dict[str, Any]:
        return {"nodes": [], "edges": []}
