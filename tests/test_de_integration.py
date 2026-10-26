import unittest
from pathlib import Path

from agents.config import AppConfig
from agents.contraindication import check_drug_contraindication
from agents.enhanced_pipeline import run_enhanced_pipeline_core
from agents.integrations.context import DeIntegrationContext
from agents.integrations.graph_client import MockGraphClient
from agents.integrations.mapping import item_seq_from_drug_id, resolve_current_item_seqs
from agents.integrations.safety_de import merge_assess_into_contraindication
from agents.integrations.types import AssessOutcome, DataBackend, DataProvenance, provenance_to_dict
from agents.integrations.vector_client import MockVectorClient
from agents.loader import load_drugs, load_symptoms
from agents.models import Drug, UserInput
from agents.retriever import retrieve_candidates


def _test_config(**overrides) -> AppConfig:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig(
        base_dir=root / "agents",
        data_dir=root / "data",
        de_root=root / "de",
    )
    fields = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
    fields.update(overrides)
    return AppConfig(**fields)


class MappingTests(unittest.TestCase):
    def test_item_seq_matches_drug_id(self):
        self.assertEqual(item_seq_from_drug_id("195700020"), "195700020")

    def test_resolve_current_from_ids(self):
        drugs = [Drug("195700020", "활명수", None, [], "", "", "", None, None, [], "", [])]
        ui = UserInput(symptoms=["소화불량"], current_drug_ids=["195700020"])
        seqs, notes = resolve_current_item_seqs(ui, drugs)
        self.assertEqual(seqs, ["195700020"])
        self.assertEqual(len(notes), 1)


class Neo4jSafetyMergeTests(unittest.TestCase):
    def test_ban_blocks_even_if_legacy_ok(self):
        outcome = AssessOutcome(
            ok=True,
            backend=DataBackend.MOCK,
            attempted_neo4j=True,
            payload={"verdict": "BAN", "findings": [{"message": "병용금기"}], "notes": []},
        )
        blocked, penalty, warnings, risk, backend, _, _ = merge_assess_into_contraindication(
            False, 0.0, [], "low", outcome, fallback_legacy=True
        )
        self.assertTrue(blocked)
        self.assertGreaterEqual(penalty, 10.0)
        self.assertEqual(risk, "high")
        self.assertEqual(backend, DataBackend.MOCK)

    def test_neo4j_failure_marks_unverified_with_fallback(self):
        outcome = AssessOutcome(
            ok=False,
            backend=DataBackend.UNAVAILABLE,
            attempted_neo4j=True,
            error="connection timed out",
        )
        blocked, penalty, warnings, risk, backend, notes, _ = merge_assess_into_contraindication(
            False, 0.0, [], "low", outcome, fallback_legacy=True
        )
        self.assertFalse(blocked)
        self.assertEqual(backend, DataBackend.UNAVAILABLE)
        self.assertTrue(any("미검증" in w for w in warnings))


class MockDePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.drugs = load_drugs(root / "data" / "drugs.json")
        cls.symptoms = load_symptoms(root / "data" / "symptoms.json")

    def _ctx(self, ban: bool = False) -> DeIntegrationContext:
        ban_pairs = [("195700020", "196900058")] if ban else []
        return DeIntegrationContext(
            vector=MockVectorClient(),
            graph=MockGraphClient(ban_pairs=ban_pairs),
            fallback_legacy=True,
            prefer_de=True,
        )

    def test_mock_postgresql_drives_retrieval(self):
        cfg = _test_config(use_de_vector=True, use_chroma=False, de_mock=False)
        ui = UserInput(symptoms=["소화불량"])
        results, backend, notes = retrieve_candidates(
            ui,
            self.drugs,
            top_k=2,
            config=cfg,
            de_ctx=self._ctx(),
            use_chroma=False,
        )
        self.assertIn(backend, (DataBackend.MOCK, DataBackend.HYBRID))
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].retrieval_source, "mock")
        self.assertIsNotNone(results[0].de_score)

    def test_neo4j_ban_excludes_recommendation(self):
        cfg = _test_config(use_de_vector=True, use_de_graph=True, use_chroma=False)
        ui = UserInput(
            symptoms=["기침"],
            current_drug_ids=["195700020"],
        )
        candidate = next(d for d in self.drugs if d.drug_id == "196900058")
        check = check_drug_contraindication(
            ui,
            candidate,
            self.drugs,
            graph_client=MockGraphClient(ban_pairs=[("195700020", "196900058")]),
            fallback_legacy=True,
        )
        self.assertTrue(check.blocked)
        self.assertEqual(check.neo4j_verdict, "BAN")

    def test_pipeline_provenance_with_mock_de(self):
        cfg = _test_config(
            use_de_vector=True,
            use_de_graph=True,
            use_chroma=False,
            de_mock=True,
        )
        ui = UserInput(symptoms=["소화불량", "체함"])
        result = run_enhanced_pipeline_core(
            ui,
            self.drugs,
            self.symptoms,
            top_k=2,
            config=cfg,
        )
        self.assertIsNotNone(result.data_provenance)
        self.assertIn(
            result.data_provenance.retrieval,
            (DataBackend.MOCK, DataBackend.HYBRID, DataBackend.UNAVAILABLE, DataBackend.LEGACY_JSON),
        )
        self.assertTrue(result.decision.recommended or result.decision.rejected)


class StrictDeModeTests(unittest.TestCase):
    def test_neo4j_failure_blocks_when_no_legacy_fallback(self):
        outcome = AssessOutcome(
            ok=False,
            backend=DataBackend.UNAVAILABLE,
            attempted_neo4j=True,
            error="dns failed",
        )
        blocked, penalty, warnings, risk, backend, _, _ = merge_assess_into_contraindication(
            False, 0.0, [], "low", outcome, fallback_legacy=False
        )
        self.assertTrue(blocked)
        self.assertEqual(backend, DataBackend.UNAVAILABLE)
        self.assertEqual(risk, "high")

    def test_provenance_serializes_for_api(self):
        p = DataProvenance(
            retrieval=DataBackend.MOCK,
            safety=DataBackend.NEO4J,
            notes=["summary retrieval=mock safety=neo4j"],
        )
        d = provenance_to_dict(p)
        self.assertEqual(d["retrieval"], "mock")
        self.assertEqual(d["safety"], "neo4j")
        self.assertIn("summary", d["notes"][0])


class LegacyRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.drugs = load_drugs(root / "data" / "drugs.json")
        cls.symptoms = load_symptoms(root / "data" / "symptoms.json")

    def test_legacy_only_when_de_disabled(self):
        cfg = _test_config(use_de_vector=False, use_de_graph=False, use_chroma=False)
        ui = UserInput(symptoms=["두통"])
        result = run_enhanced_pipeline_core(
            ui,
            self.drugs,
            self.symptoms,
            top_k=2,
            config=cfg,
        )
        self.assertEqual(result.data_provenance.retrieval, DataBackend.LEGACY_JSON)
        self.assertFalse(result.is_emergency)


if __name__ == "__main__":
    unittest.main()
