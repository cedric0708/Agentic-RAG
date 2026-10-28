from __future__ import annotations

from agents.config import AppConfig
from agents.integrations.graph_client import (
    DisabledGraphClient,
    DrugGraphClient,
    LiveGraphClient,
    MockGraphClient,
)
from agents.integrations.vector_client import (
    DisabledVectorClient,
    DrugVectorClient,
    LiveVectorClient,
    MockVectorClient,
)


def create_vector_client(config: AppConfig) -> DrugVectorClient:
    if config.de_mock:
        return MockVectorClient()
    if config.use_de_vector:
        return LiveVectorClient()
    return DisabledVectorClient()


def create_graph_client(config: AppConfig) -> DrugGraphClient:
    if config.de_mock:
        return MockGraphClient()
    if config.use_de_graph:
        return LiveGraphClient()
    return DisabledGraphClient()
