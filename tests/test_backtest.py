import json
import tempfile
import unittest
from pathlib import Path

from amber.backtest.backtester import event_backtest


class TestBacktest(unittest.TestCase):
<<<<<<< HEAD
    def test_event_backtest_raises_if_no_dataset_runs(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "No dataset_\\* directories found"):
                event_backtest(Path(td))

=======
>>>>>>> origin/main
    def test_event_backtest_runs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds = root / "dataset_1"
            ds.mkdir(parents=True)
            rows = [
                {"up_hit": 1, "down_hit": 0, "up_pct": 0.01},
                {"up_hit": 0, "down_hit": 1, "up_pct": 0.01},
                {"up_hit": 0, "down_hit": 0, "up_pct": 0.01},
            ]
            with (ds / "dataset.jsonl").open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            res = event_backtest(root)
            self.assertEqual(res["signals"], 3)
            self.assertIn("sharpe", res)
<<<<<<< HEAD
            self.assertAlmostEqual(res["resolution_rate"], 2 / 3)
            self.assertAlmostEqual(res["win_rate"], 1 / 2)
            self.assertGreaterEqual(res["profit_factor"], 0.0)
            self.assertIn("payoff_ratio", res)
            self.assertAlmostEqual(res["expectancy"], res["avg_pnl"])

    def test_event_backtest_raises_if_dataset_file_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "dataset_1").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "Missing dataset file"):
                event_backtest(root)

    def test_event_backtest_rejects_invalid_cost_params(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds = root / "dataset_1"
            ds.mkdir(parents=True)
            (ds / "dataset.jsonl").write_text('{"up_hit": 1, "down_hit": 0, "up_pct": 0.01}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-negative"):
                event_backtest(root, slippage_bps=-1, fee_bps=1)
            with self.assertRaisesRegex(ValueError, "finite numbers"):
                event_backtest(root, slippage_bps=float("nan"), fee_bps=1)

    def test_event_backtest_raises_for_invalid_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds = root / "dataset_1"
            ds.mkdir(parents=True)
            (ds / "dataset.jsonl").write_text('{"up_hit": 1}\n{bad json}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid JSONL"):
                event_backtest(root)
=======
>>>>>>> origin/main


if __name__ == "__main__":
    unittest.main()
