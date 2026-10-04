"""Ignition check, validated where the answer is known (CLAUDE.md rule 2).

Run at the real arm count (2 windows x 2 horizons, every feature as a rule):
- a market whose bursts come from nowhere must read `no_precursor`;
- a market where one feature rises a few bars before each burst must find it;
- rows that are already moving must never enter the check.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from amber.backtest.ignition import build_ignition_rows, format_report, run_ignition_check
from amber.backtest.label_sweep import _Series
from amber.models.features import MODEL_FEATURES

POSITIVE = ("precursor_found", "signal_matched_by", "single_feature_signal")


def _write_market(root: Path, *, seed: int, planted: bool, symbols: int = 6, n: int = 6000,
                  burst_p: float = 0.004) -> Path:
    """Quiet random walks with sudden 2% bursts in a random direction.

    With `planted`, `oi_roc_5` jumps two bars before each burst starts and
    stays normal otherwise — a precursor visible while price is still calm.
    Without it the bursts are independent of every feature.
    """
    rng = random.Random(seed)
    for s in range(symbols):
        d = root / "features" / f"S{s:02d}USDT"
        d.mkdir(parents=True)
        starts = {i for i in range(80, n - 40) if rng.random() < burst_p}
        drift = [0.0] * n
        warn = [False] * n
        for b in starts:
            sign = rng.choice((1, -1))
            for k in range(1, 11):
                drift[b + k] = sign * 0.002
            warn[b - 1] = warn[b] = True
        price = 100.0
        with (d / "part-000.jsonl").open("w") as fh:
            for i in range(n):
                ret = rng.gauss(0, 0.0003) + drift[i]
                price *= 1 + ret
                row = {name: rng.gauss(0, 1) for name in MODEL_FEATURES}
                if planted and warn[i]:
                    row["oi_roc_5"] = 4.0 + rng.random()
                row.update({"ts": 1_700_000_000_000 + i * 60_000, "mid_price": price, "ret_1": ret,
                            "obs": 500, "is_synthetic": False})
                fh.write(json.dumps(row) + "\n")
    return root


def _run(planted: bool, seed: int, **kw) -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = _write_market(Path(td), seed=seed, planted=planted, **kw)
        return run_ignition_check(root, max_candles_per_symbol=0, min_warmup_bars=60)


class TestIgnitionOnKnownAnswers(unittest.TestCase):
    def test_bursts_from_nowhere_are_not_predictable(self):
        for seed in (1, 2, 3):
            rep = _run(planted=False, seed=seed)
            self.assertEqual(len(rep["arms"]), 4)
            for arm in rep["arms"]:
                self.assertNotIn(arm.get("verdict", "").split(":")[0], POSITIVE, f"seed {seed}: {arm}")

    def test_too_few_moves_is_underpowered_not_no_precursor(self):
        """The first live run: ~10 moves in the test segment read as
        `no_precursor`. Even a planted precursor cannot be shown on that few,
        so the honest answer is underpowered."""
        rep = _run(planted=True, seed=1, burst_p=0.0004)
        for arm in rep["arms"]:
            if arm.get("status") == "ok":
                self.assertLess(arm["positive_episodes"], 20)
                self.assertEqual(arm["verdict"], "underpowered", format_report(rep))

    def test_a_planted_precursor_is_found_and_named(self):
        rep = _run(planted=True, seed=1)
        self.assertIn(rep["verdict"].split(":")[0], POSITIVE, format_report(rep))
        best = max((a for a in rep["arms"] if a.get("status") == "ok"),
                   key=lambda a: a["model"].get("lift_ci_low_clustered") or 0)
        named = best.get("best_rule", {}).get("rule", "") + " " + " ".join(
            t["feature"] for t in best.get("top_factors", [])[:2])
        self.assertIn("oi_roc_5", named, format_report(rep))
        self.assertIn("ИТОГ", format_report(rep))


class TestCalmFilter(unittest.TestCase):
    def _series(self, prices, synthetic=()):
        rows = [{**{f: 0.0 for f in MODEL_FEATURES}, "ts": i * 60_000, "is_synthetic": i in synthetic}
                for i in range(len(prices))]
        return {"A": _Series(rows=rows, prices=prices, ret_1=[0.0] * len(prices),
                             clean_idx=list(range(len(prices))))}

    def test_a_bar_already_moving_is_excluded(self):
        # Steady climb of 0.05%/bar: 30 bars span 1.5%, never calm.
        prices = [100 * (1.0005 ** i) for i in range(200)]
        rows, considered = build_ignition_rows(self._series(prices), window=30, horizon=15)
        self.assertGreater(considered, 0)
        self.assertEqual(rows, [])

    def test_a_quiet_bar_followed_by_a_jump_is_an_ignition(self):
        prices = [100.0] * 100 + [100.0 + 0.3 * k for k in range(1, 30)]
        rows, _ = build_ignition_rows(self._series(prices), window=30, horizon=15)
        last_calm = max(rows, key=lambda r: r["ts"])
        self.assertEqual(last_calm["move_hit"], 1)
        self.assertTrue(all(r["move_hit"] == 0 for r in rows if r["ts"] < 80 * 60_000))

    def test_a_gap_filled_stretch_is_not_read_as_calm(self):
        prices = [100.0] * 200
        rows, _ = build_ignition_rows(self._series(prices, synthetic={100}), window=30, horizon=15)
        # Bar 100 lies in the lookback or forward window of rows 84..129.
        self.assertFalse(any(84 * 60_000 <= r["ts"] <= 129 * 60_000 for r in rows))
        self.assertTrue(any(r["ts"] < 84 * 60_000 for r in rows))

    def test_the_first_bars_of_a_move_are_not_calm(self):
        """A 0.2% first bar fits inside the 0.5% window range; it must still
        be excluded, or 'predicting' it is just seeing the move begin."""
        prices = [100.0] * 100 + [100.2, 100.4] + [100.4] * 40
        rows, _ = build_ignition_rows(self._series(prices), window=30, horizon=15)
        self.assertFalse(any(r["ts"] in (100 * 60_000, 101 * 60_000) for r in rows))


if __name__ == "__main__":
    unittest.main()
