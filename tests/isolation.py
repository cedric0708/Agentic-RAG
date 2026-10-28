"""단위 테스트 간 전역 DE/E5 상태 격리 (실 모델 로드 방지)."""

from __future__ import annotations


def reset_vector_e5_cache() -> None:
    from agents.integrations import vector_client as vc

    vc.reset_shared_e5_model_cache()
