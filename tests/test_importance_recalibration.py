"""M3 (permutation importance) and M5 (rolling recalibration).

Both are diagnostics that decide real spending — whether to add order-book data,
and whether calibration has fallen behind the market regime — so they are tested
against inputs whose answer is known in advance.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from amber.models.importance import correlated_pairs, permutation_importance
from amber.models.recalibrate import calibration_error, read_dataset_tail

FEATURES = (
    "ret_1", "ret_5", "ret_20", "ret_60", "vol_z_20", "vol_ratio_20", "vol_accel",
    "oi_z_20", "oi_roc_5", "funding_z_20", "squeeze_ratio", "bb_width_20", "range_atr_14",
    "dist_to_high_20", "dist_to_low_20", "breakout_up_20", "breakout_dn_20",
    "taker_imbalance", "cvd_norm_20", "trade_count_z_20", "spread_bps",
)


def _rows(n=1500, seed=3):
    """Only vol_z_20 decides the label; cvd_norm_20 duplicates taker_imbalance."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        r = {f: rng.gauss(0, 1) for f in FEATURES}
        r["cvd_norm_20"] = r["taker_imbalance"] * 0.995 + rng.gauss(0, 0.01)
        r["up_hit"] = 1 if r["vol_z_20"] + rng.gauss(0, 0.3) > 0.5 else 0
        r["down_hit"] = 1 - r["up_hit"]
        r.update({"symbol": "BTCUSDT", "ts": 1_700_000_000_000 + i * 60_000, "horizon_steps": 15})
        out.append(r)
    return out


def _linear_model(weights):
    return {
        "model_type": "logreg_dual_v1",
        "features": list(FEATURES),
        "heads": {
            "pump": {"type": "logreg", "weights": weights, "bias": 0.0, "label_rate": 0.4},
            "dump": {"type": "logreg", "weights": {}, "bias": 0.0, "label_rate": 0.4},
        },
    }


class TestPermutationImportance(unittest.TestCase):
    def test_the_driving_feature_ranks_first(self):
        model = _linear_model({"vol_z_20": 3.0})
        res = permutation_importance(model, _rows(), n_repeats=2)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["scores"][0]["feature"], "vol_z_20")
        self.assertGreater(res["scores"][0]["importance"], 0.0)

    def test_unused_features_are_reported_useless(self):
        model = _linear_model({"vol_z_20": 3.0})
        res = permutation_importance(model, _rows(), n_repeats=2)
        self.assertIn("spread_bps", res["useless_features"])
        self.assertNotIn("vol_z_20", res["useless_features"])

    def test_importance_is_measured_not_assumed(self):
        """A model that ignores everything must show nothing as important."""
        res = permutation_importance(_linear_model({}), _rows(), n_repeats=2)
        self.assertEqual(res["carrying_count"], 0)

    def test_single_class_segment_is_reported_not_crashed(self):
        rows = _rows()
        for r in rows:
            r["up_hit"] = 0
        self.assertEqual(permutation_importance(_linear_model({"vol_z_20": 3.0}), rows)["status"], "single_class")

    def test_duplicate_features_are_flagged(self):
        pairs = correlated_pairs(_rows())
        found = {frozenset((p["a"], p["b"])) for p in pairs}
        self.assertIn(frozenset(("taker_imbalance", "cvd_norm_20")), found)


class TestCalibrationError(unittest.TestCase):
    def test_perfect_calibration_scores_near_zero(self):
        probs, labels = [], []
        for bucket in (0.1, 0.3, 0.5, 0.7, 0.9):
            for i in range(200):
                probs.append(bucket)
                labels.append(1 if i < bucket * 200 else 0)
        res = calibration_error(probs, labels)
        self.assertLess(res["ece"], 0.02)
        self.assertLess(abs(res["bias"]), 0.02)

    def test_overconfident_model_shows_positive_bias(self):
        probs = [0.9] * 500
        labels = [1 if i < 100 else 0 for i in range(500)]  # promised 90%, delivered 20%
        res = calibration_error(probs, labels)
        self.assertGreater(res["ece"], 0.5)
        self.assertGreater(res["bias"], 0.5)

    def test_underconfident_model_shows_negative_bias(self):
        probs = [0.1] * 500
        labels = [1 if i < 400 else 0 for i in range(500)]
        self.assertLess(calibration_error(probs, labels)["bias"], -0.5)

    def test_empty_input_is_safe(self):
        self.assertEqual(calibration_error([], [])["n"], 0)


class TestDatasetTail(unittest.TestCase):
    def test_reads_only_the_newest_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            d = root / "dataset_20260101T000000Z_aa"
            d.mkdir(parents=True)
            with (d / "dataset.jsonl").open("w", encoding="utf-8") as fh:
                for i in range(5000):
                    fh.write(json.dumps({"i": i}) + "\n")
            tail = read_dataset_tail(root, max_rows=100)
            self.assertEqual(len(tail), 100)
            self.assertEqual(tail[-1]["i"], 4999)  # newest kept

    def test_missing_dataset_is_not_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(read_dataset_tail(Path(td)), [])


if __name__ == "__main__":
    unittest.main()
