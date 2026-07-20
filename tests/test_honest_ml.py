"""End-to-end check that the ML pipeline trains, calibrates, evaluates and
backtests on out-of-sample time segments (no leakage between them)."""

import json
import random
import tempfile
import unittest
from pathlib import Path

from amber.backtest.backtester import event_backtest
from amber.datasets.build import build_dataset
from amber.models.calibrate import calibrate_model
from amber.models.eval import evaluate_model
from amber.models.registry import register_model
from amber.models.train import train_model


def _make_features(features_root: Path, symbol: str, n: int, seed: int) -> None:
    rng = random.Random(seed)
    feat_dir = features_root / "features" / symbol
    feat_dir.mkdir(parents=True)
    price = 100.0
    rows = []
    for i in range(n):
        ret = rng.gauss(0, 0.004)
        price *= 1 + ret
        rows.append(
            {
                "symbol": symbol,
                "ts": 1700000000000 + i * 60_000,
                "mid_price": price,
                "ret_1": ret,
                "ret_5": rng.gauss(0, 0.01),
                "ret_20": rng.gauss(0, 0.02),
                "vol_z_20": rng.gauss(0, 1),
                "oi_z_20": rng.gauss(0, 1),
                "funding_z_20": rng.gauss(0, 1),
                "spread_bps": abs(rng.gauss(2, 1)),
                "obs": i + 1,
                "is_synthetic": False,
            }
        )
    with (feat_dir / "part-000.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


class TestHonestML(unittest.TestCase):
    def test_train_calibrate_eval_backtest_out_of_sample(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            features_root = root / "features"
            datasets_root = root / "datasets"
            models_root = root / "models"

            _make_features(features_root, "BTCUSDT", 600, seed=1)
            _make_features(features_root, "ETHUSDT", 600, seed=2)

            out = build_dataset(
                features_root=features_root,
                datasets_root=datasets_root,
                symbols=["BTCUSDT", "ETHUSDT"],
                horizon_steps_list=[5, 10],
                up_pct=0.004,
                down_pct=0.004,
            )
            self.assertGreater(out["rows"], 1000)

            # dataset must be globally time-ordered (not symbol blocks)
            ds_file = sorted(datasets_root.glob("dataset_*/dataset.jsonl"))[-1]
            rows = [json.loads(x) for x in ds_file.read_text(encoding="utf-8").splitlines()]
            ts_list = [r["ts"] for r in rows]
            self.assertEqual(ts_list, sorted(ts_list))
            # censored tail rows must be dropped per horizon: each row leaves a
            # full forward window of its own horizon
            for r in rows:
                self.assertLess(r["ts"], 1700000000000 + (600 - int(r["horizon_steps"])) * 60_000)

            tr = train_model(datasets_root=datasets_root, models_root=models_root)
            self.assertTrue(tr["has_holdout_splits"])
            self.assertEqual(tr["split_mode"], "ts")
            self.assertIn(tr["model_type"], ("lightgbm_dual_v1", "logreg_dual_v1"))
            self.assertLess(tr["train_rows"], tr["total_rows"])
            register_model(models_root=models_root, model_run_id=tr["run_id"])

            model = json.loads((models_root / tr["run_id"] / "model.json").read_text(encoding="utf-8"))
            self.assertIsNotNone(model["splits"])
            self.assertIn("train_reference", model)
            self.assertIn("ret_1", model["train_reference"])
            # labeling metadata reflects real dataset params, not hardcoded 20%
            self.assertLessEqual(model["labeling"]["avg_up_pct"], 0.01)

            cal = calibrate_model(models_root=models_root, datasets_root=datasets_root)
            self.assertEqual(cal["holdout_source"], "holdout_split")

            ev = evaluate_model(models_root=models_root, datasets_root=datasets_root, threshold=0.6)
            self.assertEqual(ev["in_sample"], 0.0)
            self.assertLess(ev["rows"], ev["rows_total"])

            bt = event_backtest(
                datasets_root,
                models_root,
                thresholds={
                    "pump_prob_calibrated_min": 0.5,
                    "dump_prob_calibrated_min": 0.5,
                    "directional_score_min": 0.0,
                    "spread_bps_max": 100.0,
                },
            )
            self.assertEqual(bt["mode"], "model_signals")
            self.assertEqual(bt["segment"], "test_split")
            self.assertEqual(bt["horizon_steps"], 5)
            self.assertIn("sharpe", bt)

    def test_single_class_labels_fall_back_to_constant_head(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            datasets_root = root / "datasets"
            models_root = root / "models"
            ds = datasets_root / "dataset_flat"
            ds.mkdir(parents=True)
            rows = [
                {"ret_1": 0.0, "vol_z_20": 0.0, "up_hit": 0, "down_hit": 0}
                for _ in range(100)
            ]
            with (ds / "dataset.jsonl").open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            tr = train_model(datasets_root=datasets_root, models_root=models_root)
            model = json.loads((models_root / tr["run_id"] / "model.json").read_text(encoding="utf-8"))
            self.assertEqual(model["heads"]["pump"]["type"], "constant")
            self.assertLess(model["heads"]["pump"]["prob"], 0.01)


if __name__ == "__main__":
    unittest.main()
