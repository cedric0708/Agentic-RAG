# -*- coding: utf-8 -*-
"""manual_reasoning_trace_e2e.py CLI 분기·집계 (DB 미접속)."""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agents.config import AppConfig

import tests.manual_reasoning_trace_e2e as e2e


def _fake_probe() -> e2e.InfraProbe:
    return e2e.InfraProbe(
        postgresql_connection=True,
        neo4j_live=True,
        de_mock_env=False,
        pg_backend="postgresql",
        pg_connection_detail="mocked",
        neo4j_backend="neo4j",
        neo4j_detail="mocked",
        env_files_loaded=[],
    )


def _fake_result(name: str, status: str) -> e2e.ScenarioResult:
    return e2e.ScenarioResult(name=name, status=status, lines=[f"line for {name}"])


def _minimal_cfg() -> AppConfig:
    root = Path(__file__).resolve().parents[1]
    return AppConfig(
        base_dir=root / "agents",
        data_dir=root / "data",
        de_root=root / "de",
    )


class ResolveScenarioKeysTests(unittest.TestCase):
    def test_all(self):
        self.assertEqual(e2e.resolve_scenario_keys("all"), ["a", "b", "c"])

    def test_single(self):
        self.assertEqual(e2e.resolve_scenario_keys("b"), ["b"])
        self.assertEqual(e2e.resolve_scenario_keys(" A "), ["a"])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            e2e.resolve_scenario_keys("x")


class ExitCodeTests(unittest.TestCase):
    def test_empty_results_fail(self):
        self.assertEqual(e2e.compute_exit_code([]), 1)

    def test_skip_only_success(self):
        r = [_fake_result("A", e2e.Verdict.SKIP)]
        self.assertEqual(e2e.compute_exit_code(r), 0)

    def test_fail_returns_one(self):
        r = [_fake_result("A", e2e.Verdict.PASS), _fake_result("B", e2e.Verdict.FAIL)]
        self.assertEqual(e2e.compute_exit_code(r), 1)


class MainFlowTests(unittest.TestCase):
    def _run_main(self, *argv: str) -> tuple[int, str]:
        buf = io.StringIO()
        with redirect_stdout(buf):
            with patch.object(sys, "argv", ["manual_reasoning_trace_e2e.py", *argv]):
                code = e2e.main()
        return code, buf.getvalue()

    @patch.object(e2e, "load_symptoms", return_value=[])
    @patch.object(e2e, "load_drugs", return_value=[])
    @patch.object(e2e, "load_config", return_value=_minimal_cfg())
    @patch.object(e2e, "load_project_env", return_value=[])
    @patch.object(e2e, "probe_infrastructure", return_value=_fake_probe())
    @patch.object(e2e, "run_scenario_a")
    @patch.object(e2e, "run_scenario_b")
    @patch.object(e2e, "run_scenario_c")
    def test_scenario_a_only(self, mock_c, mock_b, mock_a, *_rest):
        mock_a.return_value = _fake_result("A. 일반 증상 검색 (한글)", e2e.Verdict.PASS)

        code, out = self._run_main("--scenario", "a")

        mock_a.assert_called_once()
        mock_b.assert_not_called()
        mock_c.assert_not_called()
        self.assertIn("실행 예정 시나리오: A", out)
        self.assertIn(">>> 시나리오 A", out)
        self.assertIn("전체 요약", out)
        self.assertIn("PASS=1", out)
        self.assertIn("FAIL=0", out)
        self.assertEqual(code, 0)

    @patch.object(e2e, "load_symptoms", return_value=[])
    @patch.object(e2e, "load_drugs", return_value=[])
    @patch.object(e2e, "load_config", return_value=_minimal_cfg())
    @patch.object(e2e, "load_project_env", return_value=[])
    @patch.object(e2e, "probe_infrastructure", return_value=_fake_probe())
    @patch.object(e2e, "run_scenario_a")
    @patch.object(e2e, "run_scenario_b")
    @patch.object(e2e, "run_scenario_c")
    def test_all_runs_three(self, mock_c, mock_b, mock_a, *_rest):
        mock_a.return_value = _fake_result("A", e2e.Verdict.PASS)
        mock_b.return_value = _fake_result("B", e2e.Verdict.SKIP)
        mock_c.return_value = _fake_result("C", e2e.Verdict.PASS)

        code, out = self._run_main("--scenario", "all")

        self.assertEqual(mock_a.call_count, 1)
        self.assertEqual(mock_b.call_count, 1)
        self.assertEqual(mock_c.call_count, 1)
        self.assertIn("PASS=2", out)
        self.assertIn("SKIP=1", out)
        self.assertEqual(code, 0)

    @patch.object(e2e, "load_symptoms", return_value=[])
    @patch.object(e2e, "load_drugs", return_value=[])
    @patch.object(e2e, "load_config", return_value=_minimal_cfg())
    @patch.object(e2e, "load_project_env", return_value=[])
    @patch.object(e2e, "probe_infrastructure", return_value=_fake_probe())
    @patch.object(e2e, "run_scenario_a")
    def test_scenario_exception_becomes_fail(self, mock_a, *_rest):
        mock_a.side_effect = RuntimeError("pipeline boom")

        code, out = self._run_main("--scenario", "a")

        self.assertIn("[FAIL]", out)
        self.assertIn("FAIL=1", out)
        self.assertEqual(code, 1)

    @patch.object(e2e, "load_symptoms", return_value=[])
    @patch.object(e2e, "load_drugs", return_value=[])
    @patch.object(e2e, "load_config", return_value=_minimal_cfg())
    @patch.object(e2e, "load_project_env", return_value=[])
    @patch.object(e2e, "probe_infrastructure", return_value=_fake_probe())
    @patch.object(e2e, "execute_scenarios", return_value=[])
    def test_no_scenario_results_exit_one(self, *_rest):
        code, out = self._run_main("--scenario", "c")
        self.assertIn("ERROR: 실행된 시나리오가 없습니다", out)
        self.assertEqual(code, 1)


class InfraProbeFieldTests(unittest.TestCase):
    def test_neo4j_detail_kw_only(self):
        """InfraProbe 생성 시 neo_detail 오타 키워드가 없어야 함."""
        probe = e2e.InfraProbe(
            postgresql_connection=False,
            neo4j_live=False,
            de_mock_env=False,
            pg_backend="x",
            pg_connection_detail="",
            neo4j_backend="x",
            neo4j_detail="ok",
            env_files_loaded=[],
        )
        self.assertEqual(probe.neo4j_detail, "ok")


if __name__ == "__main__":
    unittest.main()
