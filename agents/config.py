from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class AppConfig:
    """프로젝트 전역 설정값."""

    base_dir: Path
    data_dir: Path
    de_root: Path
    top_k: int = 3
    min_retrieval_score: float = 0.1
    clarification_min_symptoms: int = 1
    enable_agent_trace: bool = False
    enable_reasoning_trace: bool = True
    show_reasoning: bool = True
    use_chroma: bool = True
    chroma_top_n: int = 10
    chroma_score_weight: float = 5.0
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    use_de_vector: bool = True
    use_de_graph: bool = True
    de_fallback_legacy: bool = True
    de_mock: bool = False
    de_vector_score_weight: float = 8.0
    de_retrieval_top_k_multiplier: int = 3


def load_config() -> AppConfig:
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    data_dir = Path(os.getenv("DATA_DIR", project_root / "data"))
    de_root = Path(os.getenv("DE_ROOT", project_root / "database"))

    return AppConfig(
        base_dir=base_dir,
        data_dir=data_dir,
        de_root=de_root,
        top_k=int(os.getenv("TOP_K", "3")),
        min_retrieval_score=float(os.getenv("MIN_RETRIEVAL_SCORE", "0.1")),
        enable_agent_trace=os.getenv("ENABLE_AGENT_TRACE", "false").lower() in {"1", "true", "yes"},
        enable_reasoning_trace=os.getenv("ENABLE_REASONING_TRACE", "true").lower()
        in {"1", "true", "yes"},
        show_reasoning=os.getenv("SHOW_REASONING", "true").lower() in {"1", "true", "yes"},
        use_chroma=os.getenv("USE_CHROMA", "true").lower() in {"1", "true", "yes"},
        chroma_top_n=int(os.getenv("CHROMA_TOP_N", "10")),
        chroma_score_weight=float(os.getenv("CHROMA_SCORE_WEIGHT", "5.0")),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        use_de_vector=os.getenv("USE_DE_VECTOR", "true").lower() in {"1", "true", "yes"},
        use_de_graph=os.getenv("USE_DE_GRAPH", "true").lower() in {"1", "true", "yes"},
        de_fallback_legacy=os.getenv("DE_FALLBACK_LEGACY", "true").lower() in {"1", "true", "yes"},
        de_mock=os.getenv("DE_MOCK", "false").lower() in {"1", "true", "yes"},
        de_vector_score_weight=float(os.getenv("DE_VECTOR_SCORE_WEIGHT", "8.0")),
        de_retrieval_top_k_multiplier=int(os.getenv("DE_RETRIEVAL_TOP_K_MULTIPLIER", "3")),
    )
