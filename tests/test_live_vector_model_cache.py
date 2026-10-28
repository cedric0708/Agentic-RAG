# -*- coding: utf-8 -*-
"""LiveVectorClient E5 공유 캐시 (Mock loader, 실 모델 로드 없음)."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import MagicMock, patch

from agents.integrations.factory import create_vector_client
from agents.config import AppConfig
from agents.integrations.types import DataBackend
from agents.integrations import vector_client as vc
from agents.integrations.vector_client import LiveVectorClient, MockVectorClient


def _cfg(**kw) -> AppConfig:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    base = AppConfig(
        base_dir=root / "agents",
        data_dir=root / "data",
        de_root=root / "de",
    )
    fields = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
    fields.update(kw)
    return AppConfig(**fields)


class SharedE5ModelCacheTests(unittest.TestCase):
    def setUp(self):
        vc.reset_shared_e5_model_cache()

    def tearDown(self):
        vc.reset_shared_e5_model_cache()

    def test_get_shared_model_called_once(self):
        sentinel = object()
        loader = MagicMock(return_value=sentinel)
        vc.set_e5_model_loader_for_tests(loader)

        m1 = vc.get_shared_e5_model()
        m2 = vc.get_shared_e5_model()
        self.assertIs(m1, m2)
        loader.assert_called_once()

    def test_init_failure_allows_retry(self):
        loader = MagicMock(side_effect=[RuntimeError("boom"), object()])
        vc.set_e5_model_loader_for_tests(loader)

        with self.assertRaises(RuntimeError):
            vc.get_shared_e5_model()
        self.assertIsNone(vc._e5_model)

        m = vc.get_shared_e5_model()
        self.assertIsNotNone(m)
        self.assertEqual(loader.call_count, 2)

    def test_concurrent_init_single_loader_call(self):
        sentinel = object()
        calls = {"n": 0}
        count_lock = threading.Lock()

        def loader():
            with count_lock:
                calls["n"] += 1
            return sentinel

        vc.set_e5_model_loader_for_tests(loader)
        results: list = []

        def worker():
            results.append(vc.get_shared_e5_model())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual(calls["n"], 1)
        self.assertEqual(len(results), 8)
        self.assertTrue(all(r is sentinel for r in results))


class LiveVectorClientSharedModelTests(unittest.TestCase):
    def setUp(self):
        vc.reset_shared_e5_model_cache()

    def tearDown(self):
        vc.reset_shared_e5_model_cache()

    def _fake_de_module(self, shared):
        fake_db_cls = MagicMock()
        fake_mod = MagicMock()
        fake_mod.DrugVectorDB = fake_db_cls
        instance = MagicMock()
        instance.search_drugs.return_value = [{"item_seq": "1", "name": "x", "score": 0.9}]
        fake_db_cls.return_value = instance
        return fake_mod, fake_db_cls, instance

    @patch("agents.integrations.de_import.load_de_module")
    def test_two_clients_same_model_object(self, load_mod):
        shared = object()
        vc.set_e5_model_loader_for_tests(lambda: shared)
        fake_mod, fake_db_cls, _ = self._fake_de_module(shared)
        load_mod.return_value = fake_mod

        c1 = LiveVectorClient()
        c1.search_drugs("두통", top_k=1)
        c1.close()

        c2 = LiveVectorClient()
        c2.search_drugs("두통", top_k=1)

        self.assertEqual(fake_db_cls.call_count, 2)
        for call in fake_db_cls.call_args_list:
            self.assertIs(call.kwargs.get("model"), shared)

    @patch("agents.integrations.de_import.load_de_module")
    def test_repeated_search_no_extra_model_load(self, load_mod):
        shared = object()
        loader = MagicMock(return_value=shared)
        vc.set_e5_model_loader_for_tests(loader)
        fake_mod, fake_db_cls, inst = self._fake_de_module(shared)
        load_mod.return_value = fake_mod

        client = LiveVectorClient()
        client.search_drugs("a", top_k=1)
        client.search_drugs("b", top_k=1)
        loader.assert_called_once()
        inst.search_drugs.assert_called()

    @patch("agents.integrations.de_import.load_de_module")
    def test_loader_invoked_once_two_clients(self, load_mod):
        shared = object()
        loader = MagicMock(return_value=shared)
        vc.set_e5_model_loader_for_tests(loader)
        fake_mod, _, _ = self._fake_de_module(shared)
        load_mod.return_value = fake_mod

        LiveVectorClient().search_drugs("q", top_k=1)
        LiveVectorClient().search_drugs("q", top_k=1)
        loader.assert_called_once()


class MockFactoryTests(unittest.TestCase):
    def test_mock_client_does_not_touch_e5_cache(self):
        vc.reset_shared_e5_model_cache()
        loader = MagicMock(side_effect=AssertionError("should not load"))
        vc.set_e5_model_loader_for_tests(loader)

        client = create_vector_client(_cfg(de_mock=True))
        self.assertIsInstance(client, MockVectorClient)
        outcome = client.search_drugs("두통", top_k=1)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.backend, DataBackend.MOCK)
        loader.assert_not_called()


class ProvenanceRegressionTests(unittest.TestCase):
    @patch("agents.integrations.de_import.load_de_module")
    def test_search_outcome_shape_unchanged(self, load_mod):
        vc.reset_shared_e5_model_cache()
        vc.set_e5_model_loader_for_tests(lambda: object())
        inst = MagicMock()
        inst.search_drugs.return_value = [{"item_seq": "195700020", "score": 0.5}]
        fake_mod = MagicMock()
        fake_mod.DrugVectorDB = MagicMock(return_value=inst)
        load_mod.return_value = fake_mod

        outcome = LiveVectorClient().search_drugs("두통", top_k=2)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.backend, DataBackend.POSTGRESQL)
        self.assertTrue(outcome.attempted_postgresql)
        self.assertEqual(len(outcome.rows), 1)


if __name__ == "__main__":
    unittest.main()
