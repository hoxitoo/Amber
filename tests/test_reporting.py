import json
import tempfile
import unittest
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
            (root / "raw" / "ticks" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "ticks" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")
            (root / "logs").mkdir(parents=True, exist_ok=True)
            (root / "logs" / "signals.jsonl").write_text(json.dumps({"prob_up_calibrated": 0.6, "prob_down_calibrated": 0.4}) + "\n", encoding="utf-8")

            rep = build_system_report(storage)
            self.assertIn("health", rep)
            self.assertIn("quality", rep)
            self.assertIn("backtest", rep)
            self.assertIn("backtest_ok", rep)
            self.assertIn("overall_ok", rep)
            self.assertIn("overall_reason", rep)




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
            (root / "raw" / "ticks" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "ticks" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
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
            (root / "raw" / "ticks" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "ticks" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
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
            (root / "raw" / "ticks" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "ticks" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
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
            (root / "raw" / "ticks" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "ticks" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")
            (root / "logs").mkdir(parents=True, exist_ok=True)
            (root / "logs" / "signals.jsonl").write_text(json.dumps({"prob_up_calibrated": 0.7, "prob_down_calibrated": 0.3}) + "\n", encoding="utf-8")

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
            (root / "raw" / "ticks" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "ticks" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")

            rep = build_system_report(storage)
            self.assertFalse(rep["overall_ok"])
            self.assertEqual(rep["overall_reason"], "backtest")



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
            self.assertEqual(rep["overall_reason"], "health_and_backtest")


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
            (root / "raw" / "ticks" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "raw" / "ticks" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "features" / "features" / "BTCUSDT").mkdir(parents=True, exist_ok=True)
            (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").write_text(json.dumps({"ts": 1}) + "\n", encoding="utf-8")
            (root / "models" / "model_x").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_x" / "model.json").write_text("{}", encoding="utf-8")

            rep = build_system_report(storage)
            self.assertIn("artifacts", rep)
            self.assertTrue(rep["artifacts"]["model_ready"])


    def test_build_system_report_overall_reason_health(self):
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
            self.assertEqual(rep["overall_reason"], "health")


if __name__ == "__main__":
    unittest.main()
