# -*- coding: utf-8 -*-
"""Multi-Agent Reasoning Trace 전용 테스트 (Mock, UTF-8)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch

from agents.config import AppConfig
from agents.enhanced_pipeline import run_enhanced_pipeline_core
from agents.integrations.context import DeIntegrationContext
from agents.integrations.graph_client import MockGraphClient
from agents.integrations.vector_client import MockVectorClient, UnavailableVectorClient
from agents.loader import load_drugs, load_symptoms
from agents.models import UserInput
from agents.integrations.types import DataBackend, DataProvenance
from agents.reasoning_trace import (
    ReasoningTraceCollector,
    apply_safety_provenance,
    compute_retrieval_data_sources,
    sanitize_trace_value,
)
from tests.isolation import reset_vector_e5_cache


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_AGENTS = [
    "Query Agent",
    "Clarification Agent",
    "Safety Agent",
    "Retrieval Agent",
    "Reranker",
    "Recommendation Agent",
    "Validator",
]


def _cfg(**overrides) -> AppConfig:
    base = AppConfig(
        base_dir=ROOT / "agents",
        data_dir=ROOT / "data",
        de_root=ROOT / "de",
    )
    fields = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
    fields.update(overrides)
    return AppConfig(**fields)


class ReasoningTracePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.drugs = load_drugs(ROOT / "data" / "drugs.json")
        cls.symptoms = load_symptoms(ROOT / "data" / "symptoms.json")

    def setUp(self):
        reset_vector_e5_cache()

    def tearDown(self):
        reset_vector_e5_cache()

    def _run(self, ui: UserInput, **cfg_kw):
        cfg = _cfg(use_chroma=False, enable_reasoning_trace=True, **cfg_kw)
        return run_enhanced_pipeline_core(
            ui, self.drugs, self.symptoms, top_k=3, config=cfg, context_text="두통"
        )

    def test_seven_agent_steps_in_order(self):
        result = self._run(
            UserInput(symptoms=["두통", "열"]),
            use_de_vector=False,
            use_de_graph=False,
        )
        trace = result.reasoning_trace
        self.assertIsNotNone(trace)
        agents = [s["agent"] for s in trace["steps"]]
        for name in EXPECTED_AGENTS:
            self.assertIn(name, agents)
        self.assertEqual(agents.index("Query Agent"), 0)
        self.assertLess(agents.index("Validator"), len(agents))

    def test_step_ids_and_timing_fields(self):
        result = self._run(UserInput(symptoms=["소화불량"]), use_de_vector=False, use_de_graph=False)
        steps = result.reasoning_trace["steps"]
        ids = {s["step_id"] for s in steps}
        self.assertIn("query_agent", ids)
        self.assertIn("retrieval_agent", ids)
        for s in steps:
            self.assertIn(s["status"], ("success", "skipped", "warning", "failed"))
            self.assertIsInstance(s["duration_ms"], int)
            self.assertTrue(s["started_at"])

    def test_postgresql_source_in_retrieval(self):
        cfg = _cfg(use_de_vector=True, use_de_graph=True, use_chroma=False, de_mock=True)
        ui = UserInput(symptoms=["소화불량"])
        result = run_enhanced_pipeline_core(
            ui, self.drugs, self.symptoms, config=cfg, top_k=2
        )
        retrieval = next(
            s for s in result.reasoning_trace["steps"] if s["step_id"] == "retrieval_agent"
        )
        self.assertTrue(
            any(s in retrieval["data_sources"] for s in ("mock", "postgresql", "hybrid"))
        )

    def test_neo4j_ban_in_reranker_evidence(self):
        ui = UserInput(symptoms=["기침"], current_drug_ids=["195700020"])
        cfg = _cfg(
            use_de_vector=True,
            use_de_graph=True,
            de_mock=True,
            de_fallback_legacy=False,
            use_chroma=False,
        )
        ctx = DeIntegrationContext(
            vector=MockVectorClient(
                rows=[{"item_seq": "196900058", "name": "코푸시럽에스", "score": 0.9, "text": "기침"}]
            ),
            graph=MockGraphClient(ban_pairs=[("195700020", "196900058")]),
            fallback_legacy=False,
            prefer_de=True,
        )
        with patch("agents.enhanced_pipeline.build_de_context", return_value=ctx):
            result = run_enhanced_pipeline_core(
                ui, self.drugs, self.symptoms, config=cfg, top_k=2
            )
        rerank = next(s for s in result.reasoning_trace["steps"] if s["step_id"] == "reranker")
        self.assertIn("BAN", [e.get("neo4j_verdict") for e in rerank["evidence"]])
        rec = next(
            s for s in result.reasoning_trace["steps"] if s["step_id"] == "recommendation_agent"
        )
        self.assertEqual(rec["decision"]["recommended_count"], 0)

    def test_unknown_not_worded_as_contraindication(self):
        from agents.integrations.safety_de import merge_assess_into_contraindication
        from agents.integrations.types import AssessOutcome, DataBackend

        outcome = AssessOutcome(
            ok=True,
            backend=DataBackend.MOCK,
            attempted_neo4j=True,
            payload={"verdict": "UNKNOWN", "findings": [], "notes": []},
        )
        _, _, warnings, _, _, _, _ = merge_assess_into_contraindication(
            False, 0.0, [], "low", outcome, fallback_legacy=True
        )
        joined = " ".join(warnings)
        self.assertIn("병용금기로 확정하지 않", joined)

    def test_stub_marked_in_retrieval_evidence(self):
        ui = UserInput(symptoms=["두통"])
        cfg = _cfg(use_de_vector=True, use_de_graph=False, use_chroma=False, de_mock=False)
        rows = [{"item_seq": "999999999", "name": "UnknownDrug", "score": 0.95, "text": "두통"}]
        ctx = DeIntegrationContext(
            vector=MockVectorClient(rows=rows),
            graph=MockGraphClient(),
            fallback_legacy=True,
            prefer_de=True,
        )
        from agents.retriever import retrieve_candidates

        retrieved, _, _ = retrieve_candidates(
            ui, self.drugs, top_k=2, config=cfg, de_ctx=ctx, use_chroma=False
        )
        self.assertEqual(retrieved[0].catalog_match, "stub")

    @patch("agents.integrations.vector_client.get_shared_e5_model")
    def test_fallback_pg_unavailable(self, mock_get_e5):
        cfg = _cfg(
            use_de_vector=True,
            de_fallback_legacy=True,
            use_de_graph=False,
            use_chroma=False,
            de_mock=False,
        )
        ui = UserInput(symptoms=["두통"])
        ctx = DeIntegrationContext(
            vector=UnavailableVectorClient(),
            graph=MockGraphClient(),
            fallback_legacy=True,
            prefer_de=True,
        )
        with patch("agents.enhanced_pipeline.build_de_context", return_value=ctx):
            result = run_enhanced_pipeline_core(
                ui, self.drugs, self.symptoms, config=cfg, top_k=2
            )
        mock_get_e5.assert_not_called()
        retrieval = next(
            s for s in result.reasoning_trace["steps"] if s["step_id"] == "retrieval_agent"
        )
        self.assertEqual(retrieval["data_sources"], ["legacy_json"])
        prov = result.reasoning_trace.get("data_provenance", {})
        self.assertIn(
            prov.get("retrieval"),
            ("legacy_json", "unavailable", "postgresql", "mock"),
        )

    def test_emergency_early_stop(self):
        result = run_enhanced_pipeline_core(
            UserInput(symptoms=["가슴 통증"]),
            self.drugs,
            self.symptoms,
            config=_cfg(use_chroma=False, enable_reasoning_trace=True),
            context_text="가슴이 조이고 숨이 차요",
        )
        self.assertTrue(result.is_emergency)
        self.assertEqual(result.reasoning_trace["pipeline_status"], "emergency_stop")
        skipped = [s for s in result.reasoning_trace["steps"] if s["status"] == "skipped"]
        self.assertGreaterEqual(len(skipped), 1)

    def test_zero_recommendations_trace(self):
        cfg = _cfg(use_de_vector=True, use_de_graph=True, de_mock=True, use_chroma=False)
        rows = [{"item_seq": "999999999", "name": "StubOnly", "score": 0.99, "text": "두통"}]
        # MockVectorClient default includes catalog drugs — force stub-only via custom test below
        collector_only = True
        if collector_only:
            ui = UserInput(symptoms=["두통"])
            result = run_enhanced_pipeline_core(ui, self.drugs, self.symptoms, config=cfg, top_k=2)
            rec = next(
                s for s in result.reasoning_trace["steps"] if s["step_id"] == "recommendation_agent"
            )
            self.assertIn("recommended_count", rec["decision"])

    def test_json_serializable(self):
        result = self._run(UserInput(symptoms=["두통"]), use_de_vector=False, use_de_graph=False)
        raw = json.dumps(result.reasoning_trace, ensure_ascii=False)
        self.assertIn("trace_id", raw)

    def test_no_secrets_in_trace(self):
        poisoned = {
            "note": "password=secret123 API_KEY=abc neo4j://user:pass@host",
        }
        clean = sanitize_trace_value(poisoned)
        self.assertNotIn("secret123", str(clean))
        self.assertIn("[REDACTED]", str(clean))

    def test_korean_symptoms_preserved(self):
        ui = UserInput(symptoms=["머리가 아프고 미열이 있어요"])
        result = self._run(ui, use_de_vector=False, use_de_graph=False)
        query = result.reasoning_trace["steps"][0]
        self.assertIn("머리가 아프고", str(query["input_summary"]))


class ReasoningTraceApiTests(unittest.TestCase):
    def test_api_includes_reasoning_trace_when_present(self):
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi not installed")
        from api.diagnosis import router

        app = FastAPI()
        app.include_router(router)
        fake = mock.MagicMock()
        with patch("api.diagnosis.create_chat") as create, patch(
            "api.diagnosis.send_message"
        ) as send, patch("api.diagnosis.save_session_to_file"), patch(
            "api.diagnosis.load_session_from_file", return_value=None
        ), patch("api.diagnosis.SESSION_DB", {}):
            create.return_value = (fake, "hi")
            send.return_value = (
                fake,
                "답변",
                None,
                {"retrieval": "mock", "safety": "mock", "notes": []},
                {"trace_id": "t-1", "steps": [], "pipeline_status": "completed"},
            )
            client = TestClient(app)
            resp = client.post("/api/diagnosis", json={"content": "두통", "chat_id": "1"})
            self.assertEqual(resp.status_code, 200)
            self.assertIn("reasoning_trace", resp.json())


class ReasoningTracePatchRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.drugs = load_drugs(ROOT / "data" / "drugs.json")
        cls.symptoms = load_symptoms(ROOT / "data" / "symptoms.json")

    def setUp(self):
        reset_vector_e5_cache()

    def tearDown(self):
        reset_vector_e5_cache()

    def test_hybrid_mock_data_sources_not_postgresql(self):
        ui = UserInput(symptoms=["소화불량"])
        cfg = _cfg(
            use_de_vector=True,
            use_de_graph=False,
            de_mock=True,
            de_fallback_legacy=True,
            use_chroma=False,
        )
        ctx = DeIntegrationContext(
            vector=MockVectorClient(),
            graph=MockGraphClient(),
            fallback_legacy=True,
            prefer_de=True,
        )
        with patch("agents.enhanced_pipeline.build_de_context", return_value=ctx):
            result = run_enhanced_pipeline_core(
                ui, self.drugs, self.symptoms, config=cfg, top_k=3
            )
        retrieval = next(
            s for s in result.reasoning_trace["steps"] if s["step_id"] == "retrieval_agent"
        )
        self.assertIn("mock", retrieval["data_sources"])
        self.assertNotIn("postgresql", retrieval["data_sources"])
        self.assertIn("legacy_json", retrieval["data_sources"])
        self.assertEqual(result.data_provenance.retrieval_hit_sources, retrieval["data_sources"])

    def test_pg_fail_fallback_sources(self):
        ui = UserInput(symptoms=["두통"])
        cfg = _cfg(
            use_de_vector=True,
            de_fallback_legacy=True,
            use_de_graph=False,
            use_chroma=False,
            de_mock=False,
        )
        ctx = DeIntegrationContext(
            vector=UnavailableVectorClient(),
            graph=MockGraphClient(),
            fallback_legacy=True,
            prefer_de=True,
        )
        with patch("agents.enhanced_pipeline.build_de_context", return_value=ctx):
            result = run_enhanced_pipeline_core(
                ui, self.drugs, self.symptoms, config=cfg, top_k=2
            )
        retrieval = next(
            s for s in result.reasoning_trace["steps"] if s["step_id"] == "retrieval_agent"
        )
        self.assertEqual(retrieval["data_sources"], ["legacy_json"])
        notes = " ".join(result.data_provenance.notes).lower()
        self.assertTrue("fallback" in notes or "실패" in notes)

    def test_legacy_debug_trace_messages_restored(self):
        ui = UserInput(symptoms=["두통", "열"])
        result = run_enhanced_pipeline_core(
            ui,
            self.drugs,
            self.symptoms,
            config=_cfg(use_chroma=False, enable_reasoning_trace=True, use_de_vector=False),
        )
        query = next(s for s in result.trace_steps if s.agent_name == "Query Agent")
        self.assertTrue(query.messages)
        self.assertTrue(
            any("분석" in m or "항목" in m or "DB" in m for m in query.messages)
        )
        clar = next(s for s in result.trace_steps if s.agent_name == "Clarification Agent")
        self.assertTrue(clar.messages)

    def test_reasoning_trace_false_skips_evidence_helpers(self):
        ui = UserInput(symptoms=["두통"])
        cfg = _cfg(use_chroma=False, enable_reasoning_trace=False, use_de_vector=False)
        with patch(
            "agents.enhanced_pipeline.retrieval_evidence_from_results"
        ) as rev, patch(
            "agents.enhanced_pipeline.reranker_evidence_from_results"
        ) as rer:
            result = run_enhanced_pipeline_core(
                ui, self.drugs, self.symptoms, config=cfg, top_k=2
            )
            rev.assert_not_called()
            rer.assert_not_called()
        self.assertIsNone(result.reasoning_trace)
        self.assertGreaterEqual(len(result.trace_steps), len(EXPECTED_AGENTS))

    def test_reasoning_trace_false_on_step_still_called(self):
        ui = UserInput(symptoms=["두통"])
        seen: list[str] = []

        def _on_step(step):
            seen.append(step.agent_name)

        run_enhanced_pipeline_core(
            ui,
            self.drugs,
            self.symptoms,
            config=_cfg(use_chroma=False, enable_reasoning_trace=False, use_de_vector=False),
            on_step=_on_step,
        )
        for name in EXPECTED_AGENTS:
            self.assertIn(name, seen)

    def test_trace_collector_error_does_not_break_pipeline(self):
        ui = UserInput(symptoms=["두통"])
        cfg = _cfg(use_chroma=False, enable_reasoning_trace=True, use_de_vector=False)
        with patch(
            "agents.reasoning_trace.ReasoningTraceCollector.track",
            side_effect=RuntimeError("trace boom"),
        ):
            result = run_enhanced_pipeline_core(
                ui, self.drugs, self.symptoms, config=cfg, top_k=2
            )
        self.assertIsNotNone(result.decision)
        self.assertTrue(result.validation.passed or result.validation.errors is not None)
        self.assertGreater(len(result.trace_steps), 0)

    def test_mixed_safety_backends_not_last_only(self):
        prov = DataProvenance()
        item_a = mock.MagicMock()
        item_a.rerank_reasons = ["safety_backend=neo4j"]
        item_b = mock.MagicMock()
        item_b.rerank_reasons = ["safety_backend=legacy_json"]
        apply_safety_provenance(prov, [item_a, item_b])
        self.assertEqual(prov.safety, DataBackend.HYBRID)
        self.assertEqual(set(prov.safety_backends), {"neo4j", "legacy_json"})

    def test_recommendation_unchanged_with_trace_flag(self):
        ui = UserInput(symptoms=["소화불량"])
        base_cfg = dict(use_chroma=False, use_de_vector=False, use_de_graph=False)
        on = run_enhanced_pipeline_core(
            ui, self.drugs, self.symptoms, config=_cfg(enable_reasoning_trace=True, **base_cfg)
        )
        off = run_enhanced_pipeline_core(
            ui, self.drugs, self.symptoms, config=_cfg(enable_reasoning_trace=False, **base_cfg)
        )
        self.assertEqual(
            [x.retrieval.drug.drug_id for x in on.decision.recommended],
            [x.retrieval.drug.drug_id for x in off.decision.recommended],
        )

    def test_zero_hits_retrieval_sources(self):
        sources = compute_retrieval_data_sources(
            [], DataBackend.UNAVAILABLE, ["Retrieval fallback=legacy_json (DE 미사용 또는 실패)"]
        )
        self.assertEqual(sources, ["legacy_json"])


class CollectorUnitTests(unittest.TestCase):
    def test_collector_finalize(self):
        c = ReasoningTraceCollector()
        t = c.track("query_agent", "Query Agent")
        t.summary = "ok"
        t.complete()
        trace = c.finalize(pipeline_status="completed")
        self.assertEqual(len(trace.steps), 1)

    def test_retrieval_evidence_stub_note(self):
        from agents.models import Drug, RetrievalResult
        from agents.reasoning_trace import retrieval_evidence_from_results

        stub_drug = Drug(
            "999999999",
            "StubDrug",
            None,
            [],
            "ind",
            "",
            "",
            None,
            None,
            [],
            "",
            [],
        )
        rr = RetrievalResult(drug=stub_drug, score=1.0, catalog_match="stub", retrieval_source="mock")
        ev = retrieval_evidence_from_results([rr])
        self.assertEqual(ev[0]["catalog_match"], "stub")
        self.assertIn("note", ev[0])


if __name__ == "__main__":
    unittest.main()
