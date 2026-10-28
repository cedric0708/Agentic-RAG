from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from agents.config import load_config
from agents.integrations.factory import create_graph_client
from agents.integrations.graph_client import LiveGraphClient
from agents.integrations.types import DataBackend
from agents.interaction_graph import MAX_DRUGS, build_interaction_graph

router = APIRouter(prefix="/api")


class DrugInteractionGraphRequest(BaseModel):
    drug_ids: List[str] = Field(default_factory=list)
    drug_names: List[str] = Field(default_factory=list)
    age: Optional[float] = Field(default=None, ge=0, le=150)
    pregnant: bool = False
    include_ingredient_nodes: bool = True
    dur_only_subgraph: bool = False

    @model_validator(mode="after")
    def at_least_one_drug(self) -> "DrugInteractionGraphRequest":
        ids = [x for x in self.drug_ids if str(x).strip()]
        names = [x for x in self.drug_names if str(x).strip()]
        if not ids and not names:
            raise ValueError("drug_ids 또는 drug_names 중 하나 이상 필요합니다.")
        if len(ids) + len(names) > MAX_DRUGS:
            raise ValueError(f"입력 약물 수는 최대 {MAX_DRUGS}개까지 허용됩니다.")
        return self


@router.post("/drug-interaction-graph")
async def post_drug_interaction_graph(
    request: Request, body: DrugInteractionGraphRequest
):
    catalog = getattr(request.app.state, "DRUGS", []) or []
    cfg = load_config()
    graph = create_graph_client(cfg)
    try:
        payload = build_interaction_graph(
            drug_ids=body.drug_ids,
            drug_names=body.drug_names,
            catalog=catalog,
            graph=graph,
            age=body.age,
            pregnant=body.pregnant,
            include_ingredient_nodes=body.include_ingredient_nodes,
            dur_only_subgraph=body.dur_only_subgraph,
        )
    finally:
        if isinstance(graph, LiveGraphClient):
            graph.close()

    if payload.get("errors") and graph.backend in (
        DataBackend.DISABLED,
        DataBackend.UNAVAILABLE,
    ):
        raise HTTPException(status_code=503, detail=payload)

    if not payload.get("ok") and payload.get("errors"):
        raise HTTPException(status_code=400, detail=payload)

    return payload
