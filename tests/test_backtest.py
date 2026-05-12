import json
import tempfile
import unittest
from pathlib import Path

from amber.backtest.backtester import event_backtest


class TestBacktest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
