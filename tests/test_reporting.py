import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from amber.monitoring.reporting import build_system_report


class TestReporting(unittest.TestCase):
    def test_build_system_report_returns_sections(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            # minimal artifacts for health + quality
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")
            (root / "logs").mkdir(parents=True, exist_ok=True)
            (root / "logs" / "signals.jsonl").write_text(json.dumps({"prob_up_calibrated": 0.6, "prob_down_calibrated": 0.4}) + "\n", encoding="utf-8")
            with (root / "logs" / "metrics.jsonl").open("w", encoding="utf-8") as fh:
                fh.write(json.dumps({"metric": "model_precision_up_at_threshold", "value": 0.7}) + "\n")
                fh.write(json.dumps({"metric": "model_precision_down_at_threshold", "value": 0.6}) + "\n")
                fh.write(json.dumps({"metric": "model_brier_up_cal", "value": 0.2}) + "\n")
                fh.write(json.dumps({"metric": "model_brier_down_cal", "value": 0.3}) + "\n")

            rep = build_system_report(storage)
            self.assertIn("health", rep)
            self.assertIn("quality", rep)
            self.assertIn("backtest", rep)
            self.assertIn("backtest_ok", rep)
            self.assertIn("overall_ok", rep)
            self.assertIn("overall_reason", rep)
            self.assertIn("overall_reasons", rep)
            self.assertIn("readiness", rep)
            self.assertIn("readiness_summary", rep)
            self.assertIn("model_eval", rep)
            self.assertIn("model_eval_age_sec", rep)
            self.assertIn("model_eval_fresh", rep)
            self.assertIn("model_eval_status", rep)
            self.assertIn("model_eval_fresh_sec", rep)
            self.assertIn("model_eval_reasons", rep)
            self.assertIn("model_eval_complete", rep)
            self.assertIn("model_eval_gate_ok", rep)
            self.assertEqual(rep["model_eval"]["model_precision_up_at_threshold"], 0.7)
            self.assertGreaterEqual(rep["model_eval_age_sec"], 0.0)
            self.assertIn(rep["model_eval_status"], {"fresh", "stale"})
            self.assertEqual(rep["model_eval_reasons"]["status"], rep["model_eval_status"])
            self.assertEqual(rep["model_eval_reasons"]["status_reason"], "metrics_fresh")
            self.assertEqual(rep["model_eval_reasons"]["gate_reason"], "ok")
            self.assertTrue(rep["model_eval_reasons"]["completeness"]["is_complete"])
            self.assertTrue(rep["model_eval_complete"])
            self.assertEqual(rep["model_eval_gate_ok"], rep["model_eval_reasons"]["gate_ok"])
            self.assertEqual(
                rep["model_eval_reasons"]["required_metrics"],
                (
                    "model_precision_up_at_threshold",
                    "model_precision_down_at_threshold",
                    "model_brier_up_cal",
                    "model_brier_down_cal",
                ),
            )
            self.assertEqual(rep["model_eval_reasons"]["required_metric_count"], 4)
            self.assertEqual(rep["overall_reasons"]["code"], rep["overall_reason"])
            self.assertTrue(rep["overall_reasons"]["model_eval_required"])
            self.assertEqual(
                rep["overall_reasons"]["model_eval_effective_failed"],
                rep["overall_reasons"]["model_eval_failed"],
            )
            self.assertEqual(rep["readiness"]["overall_ok"], rep["overall_ok"])
            self.assertEqual(rep["readiness"]["health_ok"], rep["health"]["ok"])
            self.assertEqual(rep["readiness_summary"]["has_failures"], bool(rep["readiness_failed_components"]))
            self.assertEqual(rep["readiness_summary"]["failed_count"], len(rep["readiness_failed_components"]))

    def test_build_system_report_model_eval_uses_latest_finite_values(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")
            (root / "logs").mkdir(parents=True, exist_ok=True)
            with (root / "logs" / "metrics.jsonl").open("w", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": "2026-05-26T00:00:00+00:00", "metric": "model_brier_up_cal", "value": 0.25}) + "\n")
                fh.write(json.dumps({"ts": "2026-05-26T00:01:00+00:00", "metric": "model_brier_up_cal", "value": "nan"}) + "\n")
                fh.write(json.dumps({"ts": "2026-05-26T00:02:00+00:00", "metric": "model_brier_up_cal", "value": 0.2}) + "\n")

            rep = build_system_report(storage)
            self.assertEqual(rep["model_eval"]["model_brier_up_cal"], 0.2)
            self.assertGreaterEqual(rep["model_eval_age_sec"], 0.0)
            self.assertIsInstance(rep["model_eval_fresh"], bool)
            self.assertIn(rep["model_eval_status"], {"fresh", "stale"})

    def test_build_system_report_model_eval_ts_parsing_prefers_newest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")
            (root / "logs").mkdir(parents=True, exist_ok=True)
            with (root / "logs" / "metrics.jsonl").open("w", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": "bad-ts", "metric": "model_precision_up_at_threshold", "value": 0.1}) + "\n")
                fh.write(json.dumps({"ts": "2026-05-26T00:00:00+00:00", "metric": "model_precision_up_at_threshold", "value": 0.7}) + "\n")
                fh.write(json.dumps({"ts": "2026-05-26T00:02:00", "metric": "model_precision_up_at_threshold", "value": 0.9}) + "\n")

            rep = build_system_report(storage)
            self.assertEqual(rep["model_eval"]["model_precision_up_at_threshold"], 0.9)
            self.assertGreaterEqual(rep["model_eval_age_sec"], 0.0)
            self.assertIsInstance(rep["model_eval_fresh"], bool)
            self.assertIn(rep["model_eval_status"], {"fresh", "stale"})

    def test_build_system_report_model_eval_ts_parsing_supports_z_suffix(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")
            (root / "logs").mkdir(parents=True, exist_ok=True)
            with (root / "logs" / "metrics.jsonl").open("w", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": "2026-05-26T00:00:00Z", "metric": "model_precision_up_at_threshold", "value": 0.8}) + "\n")

            rep = build_system_report(storage)
            self.assertEqual(rep["model_eval"]["model_precision_up_at_threshold"], 0.8)
            self.assertIsInstance(rep["model_eval_age_sec"], float)

    def test_build_system_report_model_eval_fresh_false_for_stale_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")
            (root / "logs").mkdir(parents=True, exist_ok=True)
            with (root / "logs" / "metrics.jsonl").open("w", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": "2024-01-01T00:00:00+00:00", "metric": "model_brier_up_cal", "value": 0.25}) + "\n")

            rep = build_system_report(storage, model_eval_fresh_sec=1)
            self.assertFalse(rep["model_eval_fresh"])
            self.assertEqual(rep["model_eval_status"], "stale")
            self.assertEqual(rep["model_eval_reasons"]["status_reason"], "metrics_stale")
            self.assertEqual(rep["model_eval_reasons"]["gate_reason"], "metrics_stale")
            self.assertEqual(rep["model_eval_fresh_sec"], 60)

    def test_build_system_report_model_eval_status_missing_without_metrics_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")
            (root / "logs").mkdir(parents=True, exist_ok=True)

            rep = build_system_report(storage)
            self.assertEqual(rep["model_eval"], {})
            self.assertIsNone(rep["model_eval_age_sec"])
            self.assertFalse(rep["model_eval_fresh"])
            self.assertEqual(rep["model_eval_status"], "missing")
            self.assertEqual(rep["model_eval_reasons"]["status_reason"], "metrics_missing")
            self.assertEqual(rep["model_eval_reasons"]["gate_reason"], "metrics_missing")
            self.assertFalse(rep["model_eval_reasons"]["has_metrics"])
            self.assertFalse(rep["model_eval_reasons"]["completeness"]["is_complete"])
            self.assertFalse(rep["model_eval_complete"])
            self.assertFalse(rep["model_eval_gate_ok"])
            self.assertFalse(rep["model_eval_reasons"]["gate_ok"])
            self.assertEqual(
                rep["model_eval_reasons"]["completeness"]["missing_metrics"],
                [
                    "model_precision_up_at_threshold",
                    "model_precision_down_at_threshold",
                    "model_brier_up_cal",
                    "model_brier_down_cal",
                ],
            )

    def test_build_system_report_metrics_without_ts_are_treated_as_fresh(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")
            (root / "logs").mkdir(parents=True, exist_ok=True)
            with (root / "logs" / "metrics.jsonl").open("w", encoding="utf-8") as fh:
                fh.write(json.dumps({"metric": "model_precision_up_at_threshold", "value": 0.7}) + "\n")

            rep = build_system_report(storage)
            self.assertEqual(rep["model_eval"]["model_precision_up_at_threshold"], 0.7)
            self.assertEqual(rep["model_eval_age_sec"], 0.0)
            self.assertTrue(rep["model_eval_fresh"])
            self.assertEqual(rep["model_eval_status"], "fresh")

    def test_build_system_report_can_ignore_model_eval_for_overall_ok(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")
            ds = root / "datasets" / "dataset_1"
            ds.mkdir(parents=True, exist_ok=True)
            (ds / "dataset.jsonl").write_text(json.dumps({"up_hit": 1, "down_hit": 0, "up_pct": 0.01}) + "\n", encoding="utf-8")

            rep = build_system_report(storage, require_model_eval_for_overall_ok=False)
            self.assertTrue(rep["backtest_ok"])
            self.assertTrue(rep["health"]["ok"])
            self.assertFalse(rep["model_eval_ok"])
            self.assertFalse(rep["model_eval_complete"])
            self.assertTrue(rep["overall_ok"])
            self.assertEqual(rep["overall_reason"], "ok")
            self.assertFalse(rep["overall_reasons"]["model_eval_required"])
            self.assertTrue(rep["overall_reasons"]["model_eval_failed"])
            self.assertTrue(rep["overall_reasons"]["model_eval_incomplete"])
            self.assertFalse(rep["overall_reasons"]["model_eval_effective_failed"])
            self.assertTrue(rep["readiness"]["overall_ok"])
            self.assertEqual(rep["readiness_failed_components"], [])

    def test_build_system_report_gate_reason_incomplete_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")
            (root / "logs").mkdir(parents=True, exist_ok=True)
            with (root / "logs" / "metrics.jsonl").open("w", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "metric": "model_precision_up_at_threshold", "value": 0.7}) + "\n")
            ds = root / "datasets" / "dataset_1"
            ds.mkdir(parents=True, exist_ok=True)
            (ds / "dataset.jsonl").write_text(json.dumps({"up_hit": 1, "down_hit": 0, "up_pct": 0.01}) + "\n", encoding="utf-8")

            rep = build_system_report(storage)
            self.assertEqual(rep["model_eval_status"], "fresh")
            self.assertFalse(rep["model_eval_complete"])
            self.assertFalse(rep["model_eval_gate_ok"])
            self.assertEqual(rep["model_eval_reasons"]["gate_reason"], "metrics_incomplete")

    def test_build_system_report_overall_reason_model_eval_when_only_eval_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")
            ds = root / "datasets" / "dataset_1"
            ds.mkdir(parents=True, exist_ok=True)
            (ds / "dataset.jsonl").write_text(json.dumps({"up_hit": 1, "down_hit": 0, "up_pct": 0.01}) + "\n", encoding="utf-8")

            rep = build_system_report(storage)
            self.assertTrue(rep["health"]["ok"])
            self.assertTrue(rep["backtest_ok"])
            self.assertFalse(rep["model_eval_ok"])
            self.assertFalse(rep["overall_ok"])
            self.assertEqual(rep["overall_reason"], "model_eval")
            self.assertEqual(rep["readiness_failed_components"], ["model_eval"])

    def test_build_system_report_readiness_failed_components_consistency(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            # Set up only health data so health passes; backtest and model_eval should fail.
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")

            rep = build_system_report(storage)
            failed = set(rep["readiness_failed_components"])
            self.assertIn("backtest", failed)
            self.assertIn("model_eval", failed)
            self.assertNotIn("health", failed)
            self.assertTrue(rep["readiness_summary"]["has_failures"])
            self.assertEqual(rep["readiness_summary"]["failed_count"], len(rep["readiness_failed_components"]))

    def test_build_system_report_reason_ignores_optional_model_eval_in_combinations(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            # Keep health failing by omitting required raw/features artifacts.
            # Keep backtest failing by not creating datasets.
            # Keep model_eval stale/missing by not creating metrics.
            (root / "logs").mkdir(parents=True, exist_ok=True)

            rep = build_system_report(storage, require_model_eval_for_overall_ok=False)
            self.assertFalse(rep["health"]["ok"])
            self.assertFalse(rep["backtest_ok"])
            self.assertFalse(rep["model_eval_ok"])
            self.assertEqual(rep["overall_reason"], "health_and_backtest")
            self.assertEqual(set(rep["readiness_failed_components"]), {"health", "backtest"})
            self.assertNotIn("model_eval", rep["readiness_failed_components"])




    def test_build_system_report_requires_storage_keys(self):
        with self.assertRaisesRegex(ValueError, "Missing storage key"):
            build_system_report({"logs_dir": "/tmp"})



    def test_build_system_report_has_generated_at(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")

            rep = build_system_report(storage)
            self.assertIn("generated_at", rep)
            from datetime import datetime
            self.assertIsNotNone(datetime.fromisoformat(rep["generated_at"]))


    def test_build_system_report_backtest_error_sets_flag(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")

            rep = build_system_report(storage)
            self.assertFalse(rep["backtest_ok"])
            self.assertIn("error", rep["backtest"])


    def test_build_system_report_overall_ok_false_on_backtest_error(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")

            rep = build_system_report(storage)
            self.assertFalse(rep["backtest_ok"])
            self.assertFalse(rep["overall_ok"])



    def test_build_system_report_overall_ok_true_when_backtest_present(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")
            (root / "logs").mkdir(parents=True, exist_ok=True)
            (root / "logs" / "signals.jsonl").write_text(json.dumps({"prob_up_calibrated": 0.7, "prob_down_calibrated": 0.3}) + "\n", encoding="utf-8")
            with (root / "logs" / "metrics.jsonl").open("w", encoding="utf-8") as fh:
                now_ts = datetime.now(timezone.utc).isoformat()
                fh.write(json.dumps({"ts": now_ts, "metric": "model_precision_up_at_threshold", "value": 0.7}) + "\n")
                fh.write(json.dumps({"ts": now_ts, "metric": "model_precision_down_at_threshold", "value": 0.6}) + "\n")
                fh.write(json.dumps({"ts": now_ts, "metric": "model_brier_up_cal", "value": 0.2}) + "\n")
                fh.write(json.dumps({"ts": now_ts, "metric": "model_brier_down_cal", "value": 0.3}) + "\n")

            ds = root / "datasets" / "dataset_1"
            ds.mkdir(parents=True, exist_ok=True)
            (ds / "dataset.jsonl").write_text(json.dumps({"up_hit": 1, "down_hit": 0, "up_pct": 0.01}) + "\n", encoding="utf-8")

            rep = build_system_report(storage)
            self.assertTrue(rep["backtest_ok"])
            self.assertTrue(rep["overall_ok"])



    def test_build_system_report_overall_reason_backtest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")

            rep = build_system_report(storage)
            self.assertFalse(rep["overall_ok"])
            self.assertEqual(rep["overall_reason"], "backtest_and_model_eval")



    def test_build_system_report_overall_reason_health_and_backtest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            # only model artifact is present; health should fail (missing raw/features freshness),
            # and backtest should fail (missing dataset), producing combined reason.
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")

            rep = build_system_report(storage)
            self.assertFalse(rep["overall_ok"])
            self.assertFalse(rep["health"]["ok"])
            self.assertFalse(rep["backtest_ok"])
            self.assertEqual(rep["overall_reason"], "health_and_backtest_and_model_eval")
            self.assertTrue(rep["overall_reasons"]["health_failed"])
            self.assertTrue(rep["overall_reasons"]["backtest_failed"])
            self.assertTrue(rep["overall_reasons"]["model_eval_failed"])


    def test_build_system_report_artifacts_model_ready_true(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "raw" / "normalized" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "normalized" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")

            rep = build_system_report(storage)
            self.assertIn("artifacts", rep)
            self.assertTrue(rep["artifacts"]["model_ready"])


    def test_build_system_report_overall_reason_health_and_model_eval(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            # backtest succeeds (dataset exists), health fails (no raw/features freshness data).
            ds = root / "datasets" / "dataset_1"
            ds.mkdir(parents=True, exist_ok=True)
            (ds / "dataset.jsonl").write_text(json.dumps({"up_hit": 1, "down_hit": 0, "up_pct": 0.01}) + "\n", encoding="utf-8")

            rep = build_system_report(storage)
            self.assertFalse(rep["overall_ok"])
            self.assertFalse(rep["health"]["ok"])
            self.assertTrue(rep["backtest_ok"])
            self.assertEqual(rep["overall_reason"], "health_and_model_eval")

    def test_build_system_report_overall_reason_health_only_when_model_eval_fresh(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            ds = root / "datasets" / "dataset_1"
            ds.mkdir(parents=True, exist_ok=True)
            (ds / "dataset.jsonl").write_text(json.dumps({"up_hit": 1, "down_hit": 0, "up_pct": 0.01}) + "\n", encoding="utf-8")
            (root / "logs").mkdir(parents=True, exist_ok=True)
            with (root / "logs" / "metrics.jsonl").open("w", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": "2099-01-01T00:00:00+00:00", "metric": "model_precision_up_at_threshold", "value": 0.9}) + "\n")
                fh.write(json.dumps({"ts": "2099-01-01T00:00:00+00:00", "metric": "model_precision_down_at_threshold", "value": 0.8}) + "\n")
                fh.write(json.dumps({"ts": "2099-01-01T00:00:00+00:00", "metric": "model_brier_up_cal", "value": 0.1}) + "\n")
                fh.write(json.dumps({"ts": "2099-01-01T00:00:00+00:00", "metric": "model_brier_down_cal", "value": 0.2}) + "\n")

            rep = build_system_report(storage)
            self.assertFalse(rep["overall_ok"])
            self.assertFalse(rep["health"]["ok"])
            self.assertTrue(rep["backtest_ok"])
            self.assertTrue(rep["model_eval_ok"])
            self.assertEqual(rep["overall_reason"], "health")


if __name__ == "__main__":
    unittest.main()
