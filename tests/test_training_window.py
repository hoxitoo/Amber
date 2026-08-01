"""The dataset must not grow with every candle ever collected.

Unbounded history made the retrain materialise ~460k row dicts (four times per
cycle: train, calibrate, eval, backtest), exhausting a 3.8 GB box and putting
the pipeline in an OOM crash loop. A rolling window bounds it permanently.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from amber.datasets.build import build_dataset
from amber.models.train import _apply_clip, _clip_bounds

SYMBOL = "BTCUSDT"
FEATURES = (
    "ret_1", "ret_5", "ret_20", "ret_60", "vol_z_20", "vol_ratio_20", "vol_accel",
    "oi_z_20", "oi_roc_5", "funding_z_20", "squeeze_ratio", "bb_width_20", "range_atr_14",
    "dist_to_high_20", "dist_to_low_20", "breakout_up_20", "breakout_dn_20",
    "taker_imbalance", "cvd_norm_20", "trade_count_z_20", "spread_bps",
)


def _write_features(root: Path, candles: int) -> None:
    rng = random.Random(5)
    d = root / "features" / SYMBOL
    d.mkdir(parents=True, exist_ok=True)
    price = 100.0
    with (d / "part-000.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(candles):
            price *= 1 + rng.gauss(0, 0.004)
            row = {f: rng.gauss(0, 1) for f in FEATURES}
            row.update({"symbol": SYMBOL, "ts": 1_700_000_000_000 + i * 60_000,
                        "mid_price": price, "obs": 240, "is_synthetic": False})
            fh.write(json.dumps(row) + "\n")


def _build(root: Path, window: int) -> tuple[int, list[dict]]:
    # Each build gets its own datasets root: run ids created within the same
    # second differ only by a random suffix, so two builds sharing a root would
    # be ordered by that suffix rather than by time.
    out_root = root / f"ds_{window}"
    out = build_dataset(
        features_root=root, datasets_root=out_root, symbols=[SYMBOL],
        horizon_steps_list=[5], up_pct=0.004, down_pct=0.004, min_warmup_bars=60,
        max_candles_per_symbol=window,
    )
    ds = sorted(out_root.glob("dataset_*/dataset.jsonl"))[-1]
    rows = [json.loads(x) for x in ds.read_text(encoding="utf-8").splitlines()]
    return out["rows"], rows


class TestRollingTrainingWindow(unittest.TestCase):
    def test_dataset_size_stops_growing_with_history(self):
        sizes = []
        for candles in (600, 1200, 2400):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                _write_features(root, candles)
                n, _ = _build(root, window=500)
                sizes.append(n)
        self.assertEqual(len(set(sizes)), 1, f"window did not bound the dataset: {sizes}")

    def test_unbounded_still_grows(self):
        """Sanity check that the test above is measuring the window, not a fluke."""
        sizes = []
        for candles in (600, 1200):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                _write_features(root, candles)
                n, _ = _build(root, window=0)
                sizes.append(n)
        self.assertLess(sizes[0], sizes[1])

    def test_window_keeps_the_most_recent_candles(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_features(root, 2000)
            _, windowed = _build(root, window=300)
            _, full = _build(root, window=0)
            self.assertEqual(max(r["ts"] for r in windowed), max(r["ts"] for r in full))
            self.assertGreater(min(r["ts"] for r in windowed), min(r["ts"] for r in full))

    def test_labels_inside_the_window_are_unchanged(self):
        """Truncating history must not alter the labels it keeps."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_features(root, 1500)
            _, windowed = _build(root, window=400)
            _, full = _build(root, window=0)
            full_by_ts = {r["ts"]: r for r in full}
            for r in windowed:
                self.assertIn(r["ts"], full_by_ts)
                self.assertEqual(r["up_hit"], full_by_ts[r["ts"]]["up_hit"])
                self.assertEqual(r["down_hit"], full_by_ts[r["ts"]]["down_hit"])


class TestInPlaceClipping(unittest.TestCase):
    def test_inplace_matches_the_copying_version(self):
        rng = random.Random(3)
        x = [[rng.gauss(0, 5) for _ in FEATURES] for _ in range(200)]
        bounds = _clip_bounds(x)
        expected = _apply_clip([row[:] for row in x], bounds)
        got = _apply_clip([row[:] for row in x], bounds, inplace=True)
        self.assertEqual(got, expected)

    def test_inplace_mutates_and_copy_does_not(self):
        rng = random.Random(4)
        x = [[rng.gauss(0, 5) for _ in FEATURES] for _ in range(200)]
        bounds = _clip_bounds(x)

        copied = [row[:] for row in x]
        _apply_clip(copied, bounds, inplace=False)
        self.assertEqual(copied, [row[:] for row in x])  # untouched

        mutated = [row[:] for row in x]
        result = _apply_clip(mutated, bounds, inplace=True)
        self.assertIs(result, mutated)


if __name__ == "__main__":
    unittest.main()
