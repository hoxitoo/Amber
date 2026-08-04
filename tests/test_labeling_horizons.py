"""Barriers must scale with the horizon, and the window must stay economic.

At the old 5-bar horizon a single absolute barrier was applied to every horizon,
so the same 0.42% was a stretch over 5 bars and easy over 20 while all rows were
pooled into one dataset. Only ~35% of trades reached a barrier; the rest expired
paying the fee, and break-even needed a 60.7% win rate against 56.0% achieved.
"""

import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from amber.datasets.build import _adaptive_threshold, build_dataset

SYMBOL = "BTCUSDT"
FEATURES = (
    "ret_1", "ret_5", "ret_20", "ret_60", "vol_z_20", "vol_ratio_20", "vol_accel",
    "oi_z_20", "oi_roc_5", "funding_z_20", "squeeze_ratio", "bb_width_20", "range_atr_14",
    "dist_to_high_20", "dist_to_low_20", "breakout_up_20", "breakout_dn_20",
    "taker_imbalance", "cvd_norm_20", "trade_count_z_20", "spread_bps",
)


def _write(root: Path, n: int, sigma: float = 0.00168) -> None:
    rng = random.Random(5)
    d = root / "features" / SYMBOL
    d.mkdir(parents=True, exist_ok=True)
    price = 100.0
    with (d / "part-000.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(n):
            ret = rng.gauss(0, sigma)
            price *= 1 + ret
            row = {f: rng.gauss(0, 1) for f in FEATURES}
            row["ret_1"] = ret
            row.update({"symbol": SYMBOL, "ts": 1_700_000_000_000 + i * 60_000,
                        "mid_price": price, "obs": 240, "is_synthetic": False})
            fh.write(json.dumps(row) + "\n")


class TestHorizonScaledBarrier(unittest.TestCase):
    def test_barrier_grows_with_sqrt_of_horizon(self):
        rets = [0.002 * ((i % 7) - 3) for i in range(200)]
        base = _adaptive_threshold(rets, 150, k=0.8, window=60, floor=0.0, cap=1.0, horizon=1)
        for h in (4, 9, 16):
            got = _adaptive_threshold(rets, 150, k=0.8, window=60, floor=0.0, cap=1.0, horizon=h)
            self.assertAlmostEqual(got, base * math.sqrt(h), places=9)

    def test_disabled_scaling_keeps_one_barrier_for_all_horizons(self):
        rets = [0.002 * ((i % 7) - 3) for i in range(200)]
        a = _adaptive_threshold(rets, 150, k=0.8, window=60, floor=0.0, cap=1.0, horizon=1)
        b = _adaptive_threshold(rets, 150, k=0.8, window=60, floor=0.0, cap=1.0, horizon=30)
        self.assertAlmostEqual(a * math.sqrt(30), b, places=9)

    def test_each_horizon_gets_its_own_barrier_in_the_dataset(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, 1200)
            build_dataset(
                features_root=root, datasets_root=root / "ds", symbols=[SYMBOL],
                horizon_steps_list=[15, 30, 60], up_pct=0.002, down_pct=0.002,
                adaptive_thresholds=True, threshold_k=0.8, threshold_vol_window=60,
                threshold_floor=0.0001, threshold_cap=0.5, min_warmup_bars=60,
                scale_threshold_by_horizon=True,
            )
            f = sorted((root / "ds").glob("dataset_*/dataset.jsonl"))[-1]
            rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]

            avg = {}
            for h in (15, 30, 60):
                hr = [r["up_pct"] for r in rows if r["horizon_steps"] == h]
                self.assertTrue(hr, f"no rows for horizon {h}")
                avg[h] = sum(hr) / len(hr)
            self.assertLess(avg[15], avg[30])
            self.assertLess(avg[30], avg[60])
            self.assertAlmostEqual(avg[60] / avg[15], math.sqrt(4), places=1)

    def test_longer_horizons_resolve_more_often(self):
        """The whole point: fewer trades expiring having paid the fee."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, 2500)
            build_dataset(
                features_root=root, datasets_root=root / "ds", symbols=[SYMBOL],
                horizon_steps_list=[5, 60], up_pct=0.002, down_pct=0.002,
                adaptive_thresholds=True, threshold_k=0.8, threshold_vol_window=60,
                threshold_floor=0.0001, threshold_cap=0.5, min_warmup_bars=60,
                scale_threshold_by_horizon=True,
            )
            f = sorted((root / "ds").glob("dataset_*/dataset.jsonl"))[-1]
            rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]

            def resolved(h):
                hr = [r for r in rows if r["horizon_steps"] == h]
                return sum(1 for r in hr if r["up_hit"] or r["down_hit"]) / len(hr)

            self.assertGreater(resolved(60), resolved(5))


class TestPlattCalibration(unittest.TestCase):
    def test_platt_output_is_continuous_unlike_isotonic_steps(self):
        from amber.signals.scorer import calibrated_prob

        cal = {"method": "platt", "a": 1.3, "b": -0.4}
        levels = {round(calibrated_prob(i / 2000, cal), 9) for i in range(1, 2000)}
        self.assertGreater(len(levels), 1500)

    def test_platt_is_monotone_so_ranking_survives(self):
        from amber.signals.scorer import calibrated_prob

        cal = {"method": "platt", "a": 1.3, "b": -0.4}
        vals = [calibrated_prob(i / 100, cal) for i in range(1, 100)]
        self.assertEqual(vals, sorted(vals))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in vals))

    def test_extreme_raw_scores_stay_in_range(self):
        from amber.signals.scorer import calibrated_prob

        cal = {"method": "platt", "a": 2.0, "b": 0.0}
        for raw in (0.0, 1.0, -0.5, 1.5):
            p = calibrated_prob(raw, cal)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)


if __name__ == "__main__":
    unittest.main()
