"""Tests for the shared training pipeline (train_app.run_training)."""

import json
import random
import tempfile
import unittest
from pathlib import Path

from amber.datasets.build import build_dataset_from_config
from amber.pipeline.train_app import NotEnoughData, run_training


class TestRunTraining(unittest.TestCase):
    def test_raises_not_enough_data_on_empty(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "datasets").mkdir()
            with self.assertRaises(NotEnoughData):
                run_training({}, root / "datasets", root / "models", root / "logs")

    def test_trains_when_dataset_present(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # build a real dataset from synthetic features
            feat = root / "features" / "features" / "BTCUSDT"
            feat.mkdir(parents=True)
            rng = random.Random(3)
            price = 100.0
            with (feat / "part-000.jsonl").open("w", encoding="utf-8") as fh:
                for i in range(400):
                    ret = rng.gauss(0, 0.01)
                    price *= 1 + ret
                    fh.write(json.dumps({
                        "symbol": "BTCUSDT", "ts": 1_700_000_000_000 + i * 60_000,
                        "mid_price": price, "ret_1": ret, "ret_60": rng.gauss(0, 0.02),
                        "vol_z_20": rng.gauss(0, 1), "bb_width_20": abs(rng.gauss(0.01, 0.005)),
                        "spread_bps": 2.0, "obs": i + 1, "is_synthetic": False,
                    }) + "\n")
            config = {
                "storage": {"features_dir": str(root / "features"), "datasets_dir": str(root / "datasets")},
                "exchange": {"bybit": {"symbols": ["BTCUSDT"]}},
                "labeling": {"horizon_steps_list": [5, 10], "up_pct": 0.01, "down_pct": 0.01, "min_warmup_bars": 60},
            }
            out = build_dataset_from_config(config)
            self.assertGreater(out["rows"], 50)

            result = run_training(config, root / "datasets", root / "models", root / "logs")
            self.assertIn("eval", result)
            self.assertIn(result["train"]["model_type"], ("lightgbm_dual_v1", "logreg_dual_v1", "constant_dual_v1"))
            # metrics file was written
            self.assertTrue((root / "logs" / "metrics.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
