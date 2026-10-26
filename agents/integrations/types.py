from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DataBackend(str, Enum):
    """실제 사용·시도한 데이터 출처 (응답/로그 구분용)."""

    POSTGRESQL = "postgresql"
    NEO4J = "neo4j"
    LEGACY_JSON = "legacy_json"
    CHROMA = "chroma"
    HYBRID = "hybrid"
    MOCK = "mock"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"


@dataclass
class VectorSearchOutcome:
    ok: bool
    rows: List[Dict[str, Any]] = field(default_factory=list)
    backend: DataBackend = DataBackend.UNAVAILABLE
    error: Optional[str] = None
    attempted_postgresql: bool = False


@dataclass
class AssessOutcome:
    ok: bool
    payload: Optional[Dict[str, Any]] = None
    backend: DataBackend = DataBackend.UNAVAILABLE
    error: Optional[str] = None
    attempted_neo4j: bool = False


@dataclass
class DataProvenance:
    """파이프라인 1회 실행의 데이터 출처 요약."""

    retrieval: DataBackend = DataBackend.LEGACY_JSON
    safety: DataBackend = DataBackend.LEGACY_JSON
    notes: List[str] = field(default_factory=list)

    def note(self, text: str) -> None:
        if text and text not in self.notes:
            self.notes.append(text)


def provenance_to_dict(provenance: Optional["DataProvenance"]) -> Optional[Dict[str, Any]]:
    """API/로그용 직렬화 (비밀정보 없음)."""
    if provenance is None:
        return None
    return {
        "retrieval": provenance.retrieval.value,
        "safety": provenance.safety.value,
        "notes": list(provenance.notes),
    }
