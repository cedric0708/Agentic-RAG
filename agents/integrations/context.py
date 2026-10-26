from __future__ import annotations

from dataclasses import dataclass

from agents.config import AppConfig
from agents.integrations.factory import create_graph_client, create_vector_client
from agents.integrations.graph_client import DrugGraphClient
from agents.integrations.vector_client import DrugVectorClient


@dataclass
class DeIntegrationContext:
    vector: DrugVectorClient
    graph: DrugGraphClient
    fallback_legacy: bool
    prefer_de: bool

    def close(self) -> None:
        self.vector.close()
        if hasattr(self.graph, "close"):
            self.graph.close()


def build_de_context(config: AppConfig) -> DeIntegrationContext:
    return DeIntegrationContext(
        vector=create_vector_client(config),
        graph=create_graph_client(config),
        fallback_legacy=config.de_fallback_legacy,
        prefer_de=config.use_de_vector or config.use_de_graph,
    )
