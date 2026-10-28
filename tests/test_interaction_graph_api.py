import unittest
from unittest.mock import patch

from agents.integrations.graph_client import MockGraphClient
from agents.loader import load_drugs
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class InteractionGraphApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.drugs = load_drugs(PROJECT_ROOT / "data" / "drugs.json")

    def _client(self):
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi not installed")
        from api.interaction_graph import router

        app = FastAPI()
        app.state.DRUGS = self.drugs
        app.include_router(router)
        return TestClient(app)

    @patch("api.interaction_graph.create_graph_client")
    def test_route_returns_graph(self, create_mock):
        create_mock.return_value = MockGraphClient(
            ban_pairs=[("200402374", "200000913")],
            status_map={"200402374": "CHECKABLE", "200000913": "CHECKABLE"},
        )
        client = self._client()
        resp = client.post(
            "/api/drug-interaction-graph",
            json={
                "drug_ids": ["200402374", "200000913"],
                "include_ingredient_nodes": True,
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertGreaterEqual(len(body["nodes"]), 2)

    @patch("api.interaction_graph.create_graph_client")
    def test_validation_empty_input(self, create_mock):
        create_mock.return_value = MockGraphClient()
        client = self._client()
        resp = client.post("/api/drug-interaction-graph", json={})
        self.assertEqual(resp.status_code, 422)

    @patch("api.interaction_graph.create_graph_client")
    def test_max_drugs_422(self, create_mock):
        create_mock.return_value = MockGraphClient()
        client = self._client()
        resp = client.post(
            "/api/drug-interaction-graph",
            json={"drug_ids": [str(i) for i in range(11)]},
        )
        self.assertEqual(resp.status_code, 422)

    @patch("api.interaction_graph.create_graph_client")
    def test_disabled_graph_503(self, create_mock):
        from agents.integrations.graph_client import DisabledGraphClient

        create_mock.return_value = DisabledGraphClient()
        client = self._client()
        resp = client.post(
            "/api/drug-interaction-graph",
            json={"drug_ids": ["200402374"]},
        )
        self.assertEqual(resp.status_code, 503)

    def test_main_app_registers_route(self):
        try:
            import main
        except ImportError:
            self.skipTest("main app import failed")
        paths = set(main.app.openapi().get("paths", {}).keys())
        self.assertIn("/api/drug-interaction-graph", paths)
        self.assertIn("/api/diagnosis", paths)


if __name__ == "__main__":
    unittest.main()
