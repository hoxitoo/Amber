"""Tests for the dashboard data helpers (pure Python, no streamlit needed)."""

import json
import tempfile
import unittest
from pathlib import Path

from amber.dashboard import data as D


class TestDashboardData(unittest.TestCase):
    def test_storage_paths_are_absolute(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = {"storage": {"raw_dir": "data/raw", "models_dir": "/abs/models"}}
            paths = D.storage_paths(config, root)
            self.assertEqual(paths["raw_dir"], str(root / "data/raw"))
            self.assertEqual(paths["models_dir"], "/abs/models")

    def test_candle_stats_counts_and_synthetic_ratio(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw"
            target = raw / "normalized" / "BTCUSDT"
            target.mkdir(parents=True)
            rows = [
                {"ts": 1000, "is_synthetic": False},
                {"ts": 2000, "is_synthetic": True},
                {"ts": 3000, "is_synthetic": False},
                "{bad json}",
            ]
            with (target / "part-000.jsonl").open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write((r if isinstance(r, str) else json.dumps(r)) + "\n")
            stats = D.candle_stats(str(raw), ["BTCUSDT", "ETHUSDT"])
            btc = next(s for s in stats if s["symbol"] == "BTCUSDT")
            self.assertEqual(btc["candles"], 3)
            self.assertAlmostEqual(btc["synthetic_pct"], 100 / 3)
            eth = next(s for s in stats if s["symbol"] == "ETHUSDT")
            self.assertEqual(eth["candles"], 0)

    def test_signal_direction_and_drivers(self):
        pump = {"prob_up_calibrated": 0.8, "prob_down_calibrated": 0.1}
        dump = {"prob_up_calibrated": 0.2, "prob_down_calibrated": 0.6}
        self.assertEqual(D.signal_direction(pump), "pump")
        self.assertEqual(D.signal_direction(dump), "dump")

        sig = {"explanation": {"top_feature_impacts": [{"pump_ret_1": 0.42}, {"dump_vol_z_20": -0.1}]}}
        drivers = D.signal_top_drivers(sig)
        self.assertIn("pump_ret_1=+0.420", drivers)
        self.assertEqual(D.signal_top_drivers({}), "—")

    def test_load_signals_most_recent_first(self):
        with tempfile.TemporaryDirectory() as td:
            logs = Path(td)
            with (logs / "signals.jsonl").open("w", encoding="utf-8") as fh:
                for i in range(5):
                    fh.write(json.dumps({"symbol": "BTCUSDT", "n": i}) + "\n")
            sigs = D.load_signals(str(logs), limit=3)
            self.assertEqual([s["n"] for s in sigs], [4, 3, 2])

    def test_safe_system_report_never_raises_on_missing_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = {
                "logs_dir": str(root / "logs"),
                "datasets_dir": str(root / "datasets"),
                "raw_dir": str(root / "raw"),
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),
            }
            report, err = D.safe_system_report(storage)
            self.assertIsNone(err)
            self.assertIn("overall_ok", report)


if __name__ == "__main__":
    unittest.main()
