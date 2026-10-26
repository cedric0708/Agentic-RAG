from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Union

from agents.integrations.types import DataBackend, VectorSearchOutcome


class DrugVectorClient(ABC):
    @property
    @abstractmethod
    def backend(self) -> DataBackend:
        ...

    @abstractmethod
    def search_drugs(self, query: Union[str, List[str]], top_k: int = 5) -> VectorSearchOutcome:
        ...

    def close(self) -> None:
        pass


class DisabledVectorClient(DrugVectorClient):
    @property
    def backend(self) -> DataBackend:
        return DataBackend.DISABLED

    def search_drugs(self, query: Union[str, List[str]], top_k: int = 5) -> VectorSearchOutcome:
        return VectorSearchOutcome(ok=False, backend=DataBackend.DISABLED, error="USE_DE_VECTOR=false")


class LiveVectorClient(DrugVectorClient):
    """de/scripts/vector_tools.DrugVectorDB 래퍼."""

    def __init__(self) -> None:
        self._db = None
        self._connect_error: str | None = None

    @property
    def backend(self) -> DataBackend:
        if self._connect_error:
            return DataBackend.UNAVAILABLE
        return DataBackend.POSTGRESQL

    def _ensure(self) -> bool:
        if self._db is not None:
            return True
        if self._connect_error:
            return False
        try:
            import os

            from agents.integrations.de_import import load_de_module

            mod = load_de_module("vector_tools")
            timeout = int(os.getenv("PG_CONNECT_TIMEOUT", "8"))
            self._db = mod.DrugVectorDB(
                dsn={
                    "host": os.getenv("PG_HOST", "localhost"),
                    "port": int(os.getenv("PG_PORT", "5432")),
                    "dbname": os.getenv("VECTOR_DB", "drug_rag"),
                    "user": os.getenv("PG_USER", "airflow"),
                    "password": os.getenv("PG_PASSWORD", "airflow"),
                    "connect_timeout": timeout,
                }
            )
            return True
        except Exception as exc:
            self._connect_error = str(exc)
            self._db = None
            return False

    def search_drugs(self, query: Union[str, List[str]], top_k: int = 5) -> VectorSearchOutcome:
        if not self._ensure():
            return VectorSearchOutcome(
                ok=False,
                backend=DataBackend.UNAVAILABLE,
                error=self._connect_error or "PostgreSQL 연결 실패",
                attempted_postgresql=True,
            )
        try:
            rows = self._db.search_drugs(query, top_k=top_k)
            return VectorSearchOutcome(
                ok=True,
                rows=list(rows or []),
                backend=DataBackend.POSTGRESQL,
                attempted_postgresql=True,
            )
        except Exception as exc:
            return VectorSearchOutcome(
                ok=False,
                backend=DataBackend.UNAVAILABLE,
                error=str(exc),
                attempted_postgresql=True,
            )

    def close(self) -> None:
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                pass
            self._db = None


class UnavailableVectorClient(DrugVectorClient):
    """연결 실패 시뮬레이션 (테스트용)."""

    def __init__(self, error: str = "connection timed out") -> None:
        self._error = error

    @property
    def backend(self) -> DataBackend:
        return DataBackend.UNAVAILABLE

    def search_drugs(self, query: Union[str, List[str]], top_k: int = 5) -> VectorSearchOutcome:
        return VectorSearchOutcome(
            ok=False,
            backend=DataBackend.UNAVAILABLE,
            error=self._error,
            attempted_postgresql=True,
        )


class MockVectorClient(DrugVectorClient):
    """단위·통합 테스트용 PostgreSQL 검색 Mock."""

    def __init__(self, rows: List[dict] | None = None) -> None:
        self._rows = rows if rows is not None else [
            {
                "item_seq": "195700020",
                "name": "활명수",
                "score": 0.88,
                "doc_type": "drug_efficacy",
                "text": "소화불량, 식욕부진",
            },
            {
                "item_seq": "196900058",
                "name": "코푸시럽에스",
                "score": 0.75,
                "doc_type": "drug_efficacy",
                "text": "기침, 가래",
            },
        ]

    @property
    def backend(self) -> DataBackend:
        return DataBackend.MOCK

    def search_drugs(self, query: Union[str, List[str]], top_k: int = 5) -> VectorSearchOutcome:
        return VectorSearchOutcome(
            ok=True,
            rows=self._rows[:top_k],
            backend=DataBackend.MOCK,
            attempted_postgresql=True,
        )
