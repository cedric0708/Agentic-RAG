from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union

from agents.integrations.types import DataBackend, VectorSearchOutcome

# DE vector_tools.MODEL_NAME 과 동일 (DE 파일 import 없이 상수만 맞춤)
E5_MODEL_NAME = "intfloat/multilingual-e5-large"

_e5_model_lock = threading.Lock()
_e5_model: Any = None
_e5_model_loader: Optional[Callable[[], Any]] = None


def _default_e5_model_loader() -> Any:
    """DE DrugVectorDB._embed()와 동일 device·모델명."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(E5_MODEL_NAME, device=device)


def get_shared_e5_model() -> Any:
    """프로세스 단일 SentenceTransformer (lazy, thread-safe)."""
    global _e5_model
    if _e5_model is not None:
        return _e5_model
    with _e5_model_lock:
        if _e5_model is not None:
            return _e5_model
        loader = _e5_model_loader or _default_e5_model_loader
        model = loader()
        _e5_model = model
        return _e5_model


def reset_shared_e5_model_cache() -> None:
    """테스트 전용: 공유 모델 캐시 초기화."""
    global _e5_model, _e5_model_loader
    with _e5_model_lock:
        _e5_model = None
        _e5_model_loader = None


def set_e5_model_loader_for_tests(loader: Optional[Callable[[], Any]]) -> None:
    """테스트 전용: 모델 생성 함수 주입."""
    global _e5_model_loader
    with _e5_model_lock:
        _e5_model_loader = loader


def probe_postgresql_liveness(*, connect_timeout: Optional[int] = None) -> tuple[bool, str]:
    """
    임베딩/E5 없이 PostgreSQL + pgvector extension 존재만 확인 (읽기 전용).
    credential은 로그에 포함하지 않음.
    """
    try:
        import psycopg2
    except ImportError:
        return False, "psycopg2 not installed"

    timeout = connect_timeout if connect_timeout is not None else int(
        os.getenv("PG_CONNECT_TIMEOUT", "8")
    )
    try:
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST", "localhost"),
            port=int(os.getenv("PG_PORT", "5432")),
            dbname=os.getenv("VECTOR_DB", "drug_rag"),
            user=os.getenv("PG_USER", "airflow"),
            password=os.getenv("PG_PASSWORD", "airflow"),
            connect_timeout=timeout,
        )
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.execute(
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                )
                row = cur.fetchone()
                if not row:
                    return False, "pgvector extension not found"
                return True, f"connection ok, pgvector={row[0]}"
        finally:
            conn.close()
    except Exception as exc:
        return False, f"{type(exc).__name__}"


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
            from agents.integrations.de_import import load_de_module

            mod = load_de_module("vector_tools")
            timeout = int(os.getenv("PG_CONNECT_TIMEOUT", "8"))
            shared_model = get_shared_e5_model()
            self._db = mod.DrugVectorDB(
                dsn={
                    "host": os.getenv("PG_HOST", "localhost"),
                    "port": int(os.getenv("PG_PORT", "5432")),
                    "dbname": os.getenv("VECTOR_DB", "drug_rag"),
                    "user": os.getenv("PG_USER", "airflow"),
                    "password": os.getenv("PG_PASSWORD", "airflow"),
                    "connect_timeout": timeout,
                },
                model=shared_model,
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
