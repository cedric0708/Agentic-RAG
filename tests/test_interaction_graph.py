import json
import unittest
from unittest.mock import patch

from agents.integrations.graph_client import LiveGraphClient, MockGraphClient
from agents.integrations.types import DataBackend
from agents.interaction_graph import (
    MAX_DRUGS,
    build_interaction_graph,
    normalize_drug_inputs,
)
from agents.loader import load_drugs
from tests.isolation import reset_vector_e5_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _mock_ban_graph() -> MockGraphClient:
    return MockGraphClient(
        ban_pairs=[("200402374", "200000913")],
        status_map={
            "200402374": "CHECKABLE",
            "200000913": "CHECKABLE",
            "999999999": "NOT_FOUND",
            "111111111": "UNKNOWN",
        },
    )


class InteractionGraphServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.drugs = load_drugs(DATA_DIR / "drugs.json")

    def tearDown(self):
        reset_vector_e5_cache()

    def test_single_drug_independent_node(self):
        g = _mock_ban_graph()
        out = build_interaction_graph(
            drug_ids=["200402374"],
            drug_names=[],
            catalog=self.drugs,
            graph=g,
            include_ingredient_nodes=False,
        )
        self.assertTrue(out["ok"])
        self.assertEqual(len(out["nodes"]), 1)
        self.assertEqual(out["nodes"][0]["type"], "Drug")
        self.assertEqual(out["edges"], [])
        self.assertEqual(out["pair_assessments"], [])

    def test_two_drugs_ban_edge(self):
        g = _mock_ban_graph()
        out = build_interaction_graph(
            drug_ids=["200402374", "200000913"],
            drug_names=[],
            catalog=self.drugs,
            graph=g,
        )
        inter = [e for e in out["edges"] if e.get("kind") == "interaction"]
        self.assertEqual(len(inter), 1)
        self.assertEqual(inter[0]["verdict"], "BAN")
        self.assertEqual(inter[0]["type"], "contraindication")
        self.assertEqual(inter[0]["evidence_source"], "mock-dur")

    def test_three_drugs_all_pairs(self):
        g = MockGraphClient(
            ban_pairs=[("A", "B"), ("A", "C")],
            status_map={"A": "CHECKABLE", "B": "CHECKABLE", "C": "CHECKABLE"},
        )
        out = build_interaction_graph(
            drug_ids=["A", "B", "C"],
            drug_names=[],
            catalog=[],
            graph=g,
        )
        inter = [e for e in out["edges"] if e.get("kind") == "interaction"]
        self.assertEqual(len(inter), 2)
        self.assertEqual(len(out["pair_assessments"]), 6)

    def test_duplicate_id_dedup(self):
        g = _mock_ban_graph()
        out = build_interaction_graph(
            drug_ids=["200402374", "200402374"],
            drug_names=[],
            catalog=self.drugs,
            graph=g,
        )
        self.assertEqual(len([n for n in out["nodes"] if n["type"] == "Drug"]), 1)

    def test_duplicate_nodes_and_edges(self):
        g = _mock_ban_graph()
        out = build_interaction_graph(
            drug_ids=["200402374", "200000913"],
            drug_names=[],
            catalog=self.drugs,
            graph=g,
        )
        node_ids = [n["id"] for n in out["nodes"]]
        self.assertEqual(len(node_ids), len(set(node_ids)))
        edge_ids = [e["id"] for e in out["edges"]]
        self.assertEqual(len(edge_ids), len(set(edge_ids)))

    def test_ambiguous_name(self):
        def multi(_q, _l):
            return [
                {"item_seq": "1", "name": "알파정"},
                {"item_seq": "2", "name": "알파캡슐"},
            ]

        g = MockGraphClient(find_drug_fn=multi)
        mapped, _ = normalize_drug_inputs([], ["알파"], catalog=[], graph=g)
        self.assertEqual(mapped[0].mapping_status, "AMBIGUOUS")

    def test_not_found_id(self):
        g = MockGraphClient(status_map={"999999999": "NOT_FOUND"})
        out = build_interaction_graph(
            drug_ids=["999999999"],
            drug_names=[],
            catalog=[],
            graph=g,
        )
        self.assertEqual(out["mapping"][0]["mapping_status"], "NOT_FOUND")

    def test_unknown_no_ingredients(self):
        g = MockGraphClient(status_map={"111111111": "UNKNOWN"})
        out = build_interaction_graph(
            drug_ids=["111111111"],
            drug_names=[],
            catalog=[],
            graph=g,
            include_ingredient_nodes=True,
        )
        self.assertEqual(out["nodes"][0]["graph_status"], "UNKNOWN")
        contains = [e for e in out["edges"] if e["type"] == "CONTAINS"]
        self.assertEqual(contains, [])

    def test_neo4j_unavailable(self):
        g = LiveGraphClient()
        g._connect_error = "mock fail"
        out = build_interaction_graph(
            drug_ids=["200402374"],
            drug_names=[],
            catalog=self.drugs,
            graph=g,
        )
        self.assertFalse(out["ok"])
        self.assertIn("Neo4j", out["errors"][0])
        self.assertEqual(out["backend"], DataBackend.UNAVAILABLE.value)

    def test_no_ok_edge_when_no_findings(self):
        g = MockGraphClient(status_map={"A": "CHECKABLE", "B": "CHECKABLE"})
        out = build_interaction_graph(
            drug_ids=["A", "B"],
            drug_names=[],
            catalog=[],
            graph=g,
        )
        inter = [e for e in out["edges"] if e.get("kind") == "interaction"]
        self.assertEqual(inter, [])
        self.assertTrue(any("OK" not in str(e.get("verdict")) for e in inter) or not inter)
        self.assertTrue(any("의미하지 않습니다" in w for w in out["warnings"]))

    def test_pregnant_assess_ban_not_interaction_edge(self):
        g = MockGraphClient(status_map={"A": "CHECKABLE", "B": "CHECKABLE"})
        out = build_interaction_graph(
            drug_ids=["A", "B"],
            drug_names=[],
            catalog=[],
            graph=g,
            pregnant=True,
        )
        inter = [e for e in out["edges"] if e.get("kind") == "interaction"]
        self.assertEqual(inter, [])
        verdicts = {p["verdict"] for p in out["pair_assessments"]}
        self.assertIn("BAN", verdicts)

    def test_evidence_preserved(self):
        g = _mock_ban_graph()
        out = build_interaction_graph(
            drug_ids=["200402374", "200000913"],
            drug_names=[],
            catalog=self.drugs,
            graph=g,
        )
        inter = [e for e in out["edges"] if e.get("kind") == "interaction"][0]
        self.assertEqual(inter["source_url"], "https://example.invalid/mock")

    def test_dur_only_subgraph_filters_non_dur_ingredients(self):
        class DurMock(MockGraphClient):
            def check_status(self, item_seq: str):
                base = super().check_status(item_seq)
                if base.get("status") == "CHECKABLE":
                    base["ingredients"] = [
                        {"code": "DUR1", "name_kor": "dur-ing", "in_dur": True},
                        {"code": "HERB1", "name_kor": "herb-ing", "in_dur": False},
                    ]
                return base

        g = DurMock(status_map={"A": "CHECKABLE"})
        full = build_interaction_graph(
            drug_ids=["A"],
            drug_names=[],
            catalog=[],
            graph=g,
            include_ingredient_nodes=True,
            dur_only_subgraph=False,
        )
        dur_only = build_interaction_graph(
            drug_ids=["A"],
            drug_names=[],
            catalog=[],
            graph=g,
            include_ingredient_nodes=True,
            dur_only_subgraph=True,
        )
        full_codes = {
            n["code"] for n in full["nodes"] if n["type"] == "Ingredient"
        }
        slim_codes = {
            n["code"] for n in dur_only["nodes"] if n["type"] == "Ingredient"
        }
        self.assertIn("DUR1", full_codes)
        self.assertIn("HERB1", full_codes)
        self.assertEqual(slim_codes, {"DUR1"})

    def test_ingredient_level_toggle(self):
        g = _mock_ban_graph()
        with_ing = build_interaction_graph(
            drug_ids=["200402374", "200000913"],
            drug_names=[],
            catalog=self.drugs,
            graph=g,
            include_ingredient_nodes=True,
        )
        without_ing = build_interaction_graph(
            drug_ids=["200402374", "200000913"],
            drug_names=[],
            catalog=self.drugs,
            graph=g,
            include_ingredient_nodes=False,
        )
        self.assertTrue(any(n["type"] == "Ingredient" for n in with_ing["nodes"]))
        self.assertFalse(any(n["type"] == "Ingredient" for n in without_ing["nodes"]))
        self.assertFalse(any(e["type"] == "CONTAINS" for e in without_ing["edges"]))

    def test_max_drugs_limit(self):
        g = MockGraphClient()
        ids = [str(i) for i in range(MAX_DRUGS + 1)]
        out = build_interaction_graph(
            drug_ids=ids,
            drug_names=[],
            catalog=[],
            graph=g,
        )
        self.assertFalse(out["ok"])
        self.assertIn(str(MAX_DRUGS), out["errors"][0])

    def test_json_serializable(self):
        g = _mock_ban_graph()
        out = build_interaction_graph(
            drug_ids=["200402374", "200000913"],
            drug_names=[],
            catalog=self.drugs,
            graph=g,
        )
        json.dumps(out, ensure_ascii=False)

    def test_schema_top_level_keys(self):
        g = _mock_ban_graph()
        out = build_interaction_graph(
            drug_ids=["200402374"],
            drug_names=[],
            catalog=self.drugs,
            graph=g,
        )
        for key in (
            "graph_id",
            "ok",
            "backend",
            "nodes",
            "edges",
            "mapping",
            "pair_assessments",
            "data_provenance",
            "warnings",
            "errors",
            "disclaimer",
        ):
            self.assertIn(key, out)


def _bidirectional_contraindication_mock() -> MockGraphClient:
    def interaction_fn(current, candidate):
        if current == ["A"]:
            return [
                {
                    "type": "contraindication",
                    "level": "ban",
                    "current_code": "D000762",
                    "candidate_code": "D000027",
                    "message": "방향 AB",
                    "source": "dur-a",
                    "source_url": "https://example.invalid/a",
                }
            ]
        return [
            {
                "type": "contraindication",
                "level": "ban",
                "current_code": "D000027",
                "candidate_code": "D000762",
                "message": "방향 BA",
                "source": "dur-b",
                "source_url": "https://example.invalid/b",
            }
        ]

    return MockGraphClient(
        status_map={"A": "CHECKABLE", "B": "CHECKABLE"},
        interaction_fn=interaction_fn,
    )


class InteractionGraphDedupTests(unittest.TestCase):
    def tearDown(self):
        reset_vector_e5_cache()

    def test_reversed_ingredient_pair_single_drug_drug_edge(self):
        g = _bidirectional_contraindication_mock()
        out = build_interaction_graph(
            drug_ids=["A", "B"],
            drug_names=[],
            catalog=[],
            graph=g,
            include_ingredient_nodes=True,
        )
        inter = [e for e in out["edges"] if e.get("kind") == "interaction"]
        self.assertEqual(len(inter), 1)
        self.assertEqual(inter[0]["ingredient_pair"], ["D000027", "D000762"])

    def test_reversed_findings_single_ingredient_ingredient_edge(self):
        g = _bidirectional_contraindication_mock()
        out = build_interaction_graph(
            drug_ids=["A", "B"],
            drug_names=[],
            catalog=[],
            graph=g,
            include_ingredient_nodes=True,
        )
        ing = [e for e in out["edges"] if e.get("type") == "CONTRAINDICATED_WITH"]
        self.assertEqual(len(ing), 1)
        self.assertEqual(ing[0]["ingredient_pair"], ["D000027", "D000762"])

    def test_distinct_ingredient_pairs_both_kept(self):
        def interaction_fn(_current, _candidate):
            return [
                {
                    "type": "contraindication",
                    "level": "ban",
                    "current_code": "D000001",
                    "candidate_code": "D000002",
                    "message": "pair1",
                    "source": "s1",
                },
                {
                    "type": "contraindication",
                    "level": "ban",
                    "current_code": "D000003",
                    "candidate_code": "D000004",
                    "message": "pair2",
                    "source": "s2",
                },
            ]

        g = MockGraphClient(
            status_map={"A": "CHECKABLE", "B": "CHECKABLE"},
            interaction_fn=interaction_fn,
        )
        out = build_interaction_graph(
            drug_ids=["A", "B"],
            drug_names=[],
            catalog=[],
            graph=g,
        )
        inter = [e for e in out["edges"] if e.get("kind") == "interaction"]
        pairs = {tuple(e["ingredient_pair"]) for e in inter}
        self.assertEqual(len(inter), 2)
        self.assertIn(("D000001", "D000002"), pairs)
        self.assertIn(("D000003", "D000004"), pairs)

    def test_same_relation_different_evidence_preserved(self):
        def interaction_fn(current, candidate):
            if current == ["A"]:
                return [
                    {
                        "type": "contraindication",
                        "level": "ban",
                        "current_code": "D000762",
                        "candidate_code": "D000027",
                        "message": "근거 A",
                        "source": "source-a",
                        "source_url": "https://example.invalid/a",
                    }
                ]
            return [
                {
                    "type": "contraindication",
                    "level": "ban",
                    "current_code": "D000027",
                    "candidate_code": "D000762",
                    "message": "근거 B",
                    "source": "source-b",
                    "source_url": "https://example.invalid/b",
                }
            ]

        g = MockGraphClient(
            status_map={"A": "CHECKABLE", "B": "CHECKABLE"},
            interaction_fn=interaction_fn,
        )
        out = build_interaction_graph(
            drug_ids=["A", "B"],
            drug_names=[],
            catalog=[],
            graph=g,
        )
        inter = [e for e in out["edges"] if e.get("kind") == "interaction"][0]
        self.assertIn("evidence_items", inter)
        self.assertEqual(len(inter["evidence_items"]), 2)
        sources = {it["evidence_source"] for it in inter["evidence_items"]}
        self.assertEqual(sources, {"source-a", "source-b"})

    def test_drug_input_order_does_not_change_undirected_edges(self):
        g = _bidirectional_contraindication_mock()
        out_ab = build_interaction_graph(
            drug_ids=["A", "B"],
            drug_names=[],
            catalog=[],
            graph=g,
        )
        out_ba = build_interaction_graph(
            drug_ids=["B", "A"],
            drug_names=[],
            catalog=[],
            graph=g,
        )
        keys_ab = {
            (e["id"], e.get("kind"), e.get("type"), tuple(e.get("ingredient_pair") or []))
            for e in out_ab["edges"]
            if e.get("kind") == "interaction"
            or e.get("type") == "CONTRAINDICATED_WITH"
        }
        keys_ba = {
            (e["id"], e.get("kind"), e.get("type"), tuple(e.get("ingredient_pair") or []))
            for e in out_ba["edges"]
            if e.get("kind") == "interaction"
            or e.get("type") == "CONTRAINDICATED_WITH"
        }
        self.assertEqual(keys_ab, keys_ba)


if __name__ == "__main__":
    unittest.main()

