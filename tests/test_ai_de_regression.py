"""AI–DE 연동 추가 점검: 판정·stub·fallback·회귀·API 호환 (Mock only)."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch

from agents.config import AppConfig
from agents.conversation_agent import ConversationalAgent
from agents.contraindication import check_drug_contraindication
from agents.enhanced_pipeline import run_enhanced_pipeline_core
from agents.integrations.context import DeIntegrationContext
from agents.integrations.graph_client import MockGraphClient
from agents.integrations.mapping import drug_from_de_row
from agents.integrations.safety_de import merge_assess_into_contraindication
from agents.integrations.types import AssessOutcome, DataBackend
from agents.integrations.vector_client import MockVectorClient, UnavailableVectorClient
from agents.loader import load_drugs, load_symptoms
from agents.models import Drug, UserInput
from agents.recommendation_agent import decide_recommendations
from agents.reranker import rerank_results
from agents.retriever import retrieve_candidates


def _cfg(**overrides) -> AppConfig:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig(
        base_dir=root / "agents",
        data_dir=root / "data",
        de_root=root / "de",
    )
    fields = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
    fields.update(overrides)
    return AppConfig(**fields)


ROOT = Path(__file__).resolve().parents[1]


class VerdictHandlingTests(unittest.TestCase):
    def _merge(self, verdict: str, findings=None):
        outcome = AssessOutcome(
            ok=True,
            backend=DataBackend.MOCK,
            attempted_neo4j=True,
            payload={
                "verdict": verdict,
                "findings": findings or [],
                "notes": [],
            },
        )
        return merge_assess_into_contraindication(
            False, 0.0, [], "low", outcome, fallback_legacy=True
        )

    def test_ban_blocks(self):
        blocked, *_ , verified = self._merge(
            "BAN", [{"type": "contraindication", "message": "병용금기"}]
        )
        self.assertTrue(blocked)
        self.assertTrue(verified)

    def test_unknown_blocks_not_labeled_contraindication(self):
        blocked, _, warnings, _, _, _, _ = self._merge("UNKNOWN")
        self.assertTrue(blocked)
        joined = " ".join(warnings)
        self.assertIn("성분 정보", joined)
        self.assertIn("병용금기로 확정하지 않", joined)

    def test_not_found_blocks(self):
        blocked, _, warnings, _, _, _, _ = self._merge("NOT_FOUND")
        self.assertTrue(blocked)
        self.assertIn("찾지 못", " ".join(warnings))

    def test_caution_not_blocked(self):
        blocked, penalty, _, risk, _, _, verified = self._merge("CAUTION")
        self.assertFalse(blocked)
        self.assertGreaterEqual(penalty, 3.0)
        self.assertEqual(risk, "medium")
        self.assertTrue(verified)

    def test_ok_not_blocked_with_disclaimer(self):
        blocked, _, warnings, _, _, _, verified = self._merge("OK")
        self.assertFalse(blocked)
        self.assertTrue(verified)
        self.assertIn("보장하지 않", " ".join(warnings))


class StubDrugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_drugs(ROOT / "data" / "drugs.json")

    def test_stub_fields(self):
        row = {
            "item_seq": "999999999",
            "name": "MockOnlyDrug",
            "score": 0.9,
            "text": "테스트 효능 문구",
        }
        drug, kind = drug_from_de_row(row, self.catalog)
        self.assertEqual(kind, "stub")
        self.assertEqual(drug.ingredient, [])
        self.assertEqual(drug.dosage, "")
        self.assertEqual(drug.warnings, "")
        self.assertEqual(drug.combination_contraindication, [])

    def test_stub_not_in_recommended(self):
        cfg = _cfg(use_de_vector=True, use_de_graph=True, use_chroma=False, de_mock=True)
        stub_rows = [
            {
                "item_seq": "999999999",
                "name": "MockOnlyDrug",
                "score": 0.95,
                "text": "두통",
            }
        ]
        ctx = DeIntegrationContext(
            vector=MockVectorClient(rows=stub_rows),
            graph=MockGraphClient(fixed_verdict={"999999999": "OK"}),
            fallback_legacy=True,
            prefer_de=True,
        )
        ui = UserInput(symptoms=["두통"])
        retrieved, backend, _ = retrieve_candidates(
            ui, self.catalog, top_k=1, config=cfg, de_ctx=ctx, use_chroma=False
        )
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0].catalog_match, "stub")
        reranked = rerank_results(
            ui, retrieved, self.catalog, graph_client=ctx.graph, de_fallback_legacy=True
        )
        decision = decide_recommendations(reranked, max_recommendations=1)
        self.assertEqual(len(decision.recommended), 0)
        self.assertTrue(any("stub" in m for m in decision.messages))


class FallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.drugs = load_drugs(ROOT / "data" / "drugs.json")
        cls.symptoms = load_symptoms(ROOT / "data" / "symptoms.json")

    def test_pg_fail_fallback_legacy_true(self):
        cfg = _cfg(use_de_vector=True, de_fallback_legacy=True, use_chroma=False)
        ctx = DeIntegrationContext(
            vector=UnavailableVectorClient(),
            graph=MockGraphClient(),
            fallback_legacy=True,
            prefer_de=True,
        )
        ui = UserInput(symptoms=["두통"])
        results, backend, notes = retrieve_candidates(
            ui, self.drugs, top_k=2, config=cfg, de_ctx=ctx, use_chroma=False
        )
        self.assertEqual(backend, DataBackend.LEGACY_JSON)
        self.assertGreater(len(results), 0)
        self.assertTrue(any("fallback" in n.lower() or "legacy" in n.lower() for n in notes))

    def test_pg_fail_fallback_legacy_false(self):
        cfg = _cfg(use_de_vector=True, de_fallback_legacy=False, use_chroma=False)
        ctx = DeIntegrationContext(
            vector=UnavailableVectorClient(),
            graph=MockGraphClient(),
            fallback_legacy=False,
            prefer_de=True,
        )
        ui = UserInput(symptoms=["두통"])
        results, backend, notes = retrieve_candidates(
            ui, self.drugs, top_k=2, config=cfg, de_ctx=ctx, use_chroma=False
        )
        self.assertEqual(len(results), 0)
        self.assertEqual(backend, DataBackend.UNAVAILABLE)

    def test_neo4j_fail_unverified_not_recommended(self):
        cfg = _cfg(use_de_vector=False, use_de_graph=True, use_chroma=False)
        ctx = DeIntegrationContext(
            vector=UnavailableVectorClient(),
            graph=MockGraphClient(fail=True),
            fallback_legacy=True,
            prefer_de=True,
        )
        ui = UserInput(symptoms=["두통"])
        # legacy retrieval only
        retrieved, _, _ = retrieve_candidates(
            ui, self.drugs, top_k=2, config=cfg, de_ctx=ctx, use_chroma=False
        )
        reranked = rerank_results(
            ui, retrieved, self.drugs, graph_client=ctx.graph, de_fallback_legacy=True
        )
        decision = decide_recommendations(reranked, max_recommendations=2)
        self.assertEqual(len(decision.recommended), 0)
        self.assertTrue(
            any("미검증" in m or "제외" in m for m in decision.messages)
            or all(item.needs_extra_verification for item in reranked)
        )


class ConversationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.drugs = load_drugs(ROOT / "data" / "drugs.json")
        cls.symptoms = load_symptoms(ROOT / "data" / "symptoms.json")

    def _agent(self) -> ConversationalAgent:
        cfg = _cfg(use_de_vector=False, use_de_graph=False, use_chroma=False)
        return ConversationalAgent(self.drugs, self.symptoms, config=cfg)

    @patch.dict(os.environ, {"USE_LLM": "false", "USE_LLM_ORCHESTRATOR": "false"}, clear=False)
    def test_clarification_before_recommendation(self):
        agent = self._agent()
        session = agent.create_session()
        r1 = agent.process_message(session, "머리가 아파요")
        self.assertIn(r1.phase.value, ("clarifying", "collecting"))
        self.assertTrue(r1.message)

    @patch.dict(os.environ, {"USE_LLM": "false", "USE_LLM_ORCHESTRATOR": "false"}, clear=False)
    def test_emergency_red_flag(self):
        agent = self._agent()
        session = agent.create_session()
        agent.process_message(session, "두통")
        r = agent.process_message(
            session,
            "갑자기 벼락처럼 머리가 터질 것 같아요",
        )
        self.assertIn(r.phase.value, ("emergency", "clarifying", "follow_up"))

    @patch.dict(os.environ, {"USE_LLM": "false", "USE_LLM_ORCHESTRATOR": "false"}, clear=False)
    def test_recommendation_with_current_meds(self):
        agent = self._agent()
        session = agent.create_session()
        agent.process_message(session, "기침")
        agent.process_message(session, "없음")
        agent.process_message(session, "195700020")
        agent.process_message(session, "없음")
        agent.process_message(session, "없음")
        r = agent.process_message(session, "추천해 주세요")
        self.assertTrue(r.message)


class PipelineRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.drugs = load_drugs(ROOT / "data" / "drugs.json")
        cls.symptoms = load_symptoms(ROOT / "data" / "symptoms.json")

    def test_general_recommendation_legacy(self):
        cfg = _cfg(use_de_vector=False, use_de_graph=False, use_chroma=False)
        ui = UserInput(symptoms=["두통", "열"])
        result = run_enhanced_pipeline_core(
            ui, self.drugs, self.symptoms, top_k=2, config=cfg
        )
        self.assertFalse(result.is_emergency)
        self.assertEqual(result.data_provenance.retrieval, DataBackend.LEGACY_JSON)

    def test_emergency_pipeline(self):
        cfg = _cfg(use_de_vector=False, use_de_graph=False, use_chroma=False)
        ui = UserInput(symptoms=["가슴 통증"])
        result = run_enhanced_pipeline_core(
            ui,
            self.drugs,
            self.symptoms,
            top_k=2,
            config=cfg,
            context_text="가슴이 조이고 숨이 차요",
        )
        self.assertTrue(result.is_emergency)


class DiagnosisApiCompatibilityTests(unittest.TestCase):
    def test_response_shape(self):
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi not installed in this environment")
        from api.diagnosis import router

        app = FastAPI()
        app.include_router(router)

        fake_bundle = mock.MagicMock()

        with patch("api.diagnosis.create_chat") as create, patch(
            "api.diagnosis.send_message"
        ) as send, patch("api.diagnosis.save_session_to_file"), patch(
            "api.diagnosis.load_session_from_file", return_value=None
        ), patch(
            "api.diagnosis.SESSION_DB", {}
        ):
            create.return_value = (fake_bundle, "welcome")
            send.return_value = (
                fake_bundle,
                "테스트 답변",
                None,
                {"retrieval": "legacy_json", "safety": "legacy_json", "notes": []},
            )
            client = TestClient(app)
            resp = client.post(
                "/api/diagnosis",
                json={"content": "안녕", "chat_id": "t1"},
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(body["role"], "ai")
            self.assertIn("content", body)
            self.assertIn("debug_trace", body)
            self.assertEqual(body["data_sources"]["retrieval"], "legacy_json")


class ClientInterfaceTests(unittest.TestCase):
    def test_mock_live_vector_same_outcome_shape(self):
        mock = MockVectorClient()
        unavail = UnavailableVectorClient()
        m = mock.search_drugs(["두통"], top_k=1)
        u = unavail.search_drugs(["두통"], top_k=1)
        self.assertTrue(hasattr(m, "ok") and hasattr(m, "backend") and hasattr(m, "rows"))
        self.assertTrue(hasattr(u, "ok") and hasattr(u, "attempted_postgresql"))

    def test_mock_graph_assess_shape(self):
        g = MockGraphClient(fixed_verdict={"123": "UNKNOWN"})
        out = g.assess([], "123")
        self.assertTrue(out.attempted_neo4j)
        self.assertEqual(out.payload["verdict"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
