"""The decomposition must tell magnitude and direction apart.

This is the tool that decides whether the project buys tick-level order flow and
L2 depth (roadmap D2/D7) or redefines itself as a volatility scanner. If it
cannot separate the two factors it would send that decision the wrong way, so
the tests build data where the right answer is known by construction:

- a volatility spike reliably precedes a 1% move  -> magnitude IS predictable
- the SIGN of that move is a fair coin            -> direction is NOT

and then the same fixture with the sign leaked into a feature, where direction
must become predictable. Anything that reports directional skill on the first
fixture is broken.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from amber.backtest.decomposition import add_factor_labels, decompose, format_report
from amber.models.features import MODEL_FEATURES

HORIZON = 15


def _write_symbol(root: Path, symbol: str, n: int, seed: int, *, directional: bool) -> None:
    """A flagged bar is followed by a burst of VOLATILITY, not of drift.

    Drift would not be a null. A drift running from an earlier spike lands in
    `ret_1`, which is a model feature, so the sign of the move already under way
    is visible at the scoring bar and direction becomes genuinely predictable —
    the fixture, not the tool, would be creating the signal. Raising the
    variance of a driftless random walk makes the magnitude predictable while
    leaving the sign a fair coin that nothing in the past can reveal.
    """
    rng = random.Random(seed)
    spikes = [rng.random() < 0.015 for _ in range(n)]
    signs = [1 if rng.random() < 0.5 else -1 for _ in range(n)]

    rows = []
    for i in range(n):
        row = {name: rng.gauss(0, 1) for name in MODEL_FEATURES}
        # the spike is visible at the scoring bar; the burst follows it
        row["vol_z_20"] = 3.0 if spikes[i] else rng.gauss(0, 1)
        row["range_atr_14"] = 2.5 if spikes[i] else rng.gauss(0, 1)
        if directional:
            # leak the coming sign into a feature the model can read
            row["ret_20"] = 2.0 * signs[i] if spikes[i] else rng.gauss(0, 1)
        rows.append(row)

    price, ts = 100.0, 1_700_000_000_000
    for i, row in enumerate(rows):
        recent = [j for j in range(max(0, i - HORIZON), i) if spikes[j]]
        inside_burst = bool(recent)
        sigma = 0.0030 if inside_burst else 0.0002
        drift = 0.0012 * signs[recent[-1]] if (inside_burst and directional) else 0.0
        row["ret_1"] = rng.gauss(0.0, sigma) + drift
        price *= 1 + row["ret_1"]
        row.update({
            "ts": ts + i * 60_000,
            "mid_price": price,
            "obs": 200,
            "is_synthetic": False,
        })

    d = root / "features" / symbol
    d.mkdir(parents=True, exist_ok=True)
    with (d / "part-000.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _fixture(root: Path, *, directional: bool, symbols: int = 4, n: int = 9000) -> None:
    """Length matters more than symbol count here: every symbol shares one time
    axis, so episodes are capped by the test segment's span in minutes divided
    by the horizon. Adding symbols adds rows but not independent observations."""
    for i in range(symbols):
        _write_symbol(root, f"S{i:02d}USDT", n, seed=700 + i, directional=directional)


class TestFactorLabels(unittest.TestCase):
    def test_move_and_direction_are_derived_from_first_touch(self):
        rows = [
            {"up_hit": 1, "down_hit": 0, "first_hit": 1},
            {"up_hit": 0, "down_hit": 1, "first_hit": -1},
            {"up_hit": 1, "down_hit": 1, "first_hit": -1},  # one-sided: dipped first
            {"up_hit": 0, "down_hit": 0, "first_hit": 0},
        ]
        self.assertEqual(add_factor_labels(rows), 0)

        self.assertEqual([r["move_hit"] for r in rows], [1, 1, 1, 0])
        self.assertEqual([r["dir_up"] for r in rows], [1, 0, 0, None])

    def test_direction_is_undefined_where_nothing_happened(self):
        """A row with no move has no direction, and must not be scored as 0."""
        rows = [{"up_hit": 0, "down_hit": 0, "first_hit": 0}]
        self.assertEqual(add_factor_labels(rows), 0)
        self.assertIsNone(rows[0]["dir_up"])

    def test_a_touch_without_first_hit_is_counted_not_swallowed(self):
        """The live bug: build_arm_rows dropped first_hit, so every dir_up came
        out None while move_hit still worked and hid it."""
        rows = [{"up_hit": 1, "down_hit": 0}]  # no first_hit at all
        self.assertEqual(add_factor_labels(rows), 1)
        self.assertEqual(rows[0]["move_hit"], 1)
        self.assertIsNone(rows[0]["dir_up"])


class TestDecompositionSeparatesTheFactors(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        _fixture(root / "coin", directional=False)
        _fixture(root / "dir", directional=True)
        cls.coin = decompose(root / "coin", horizon=HORIZON, barrier=0.010, budget=0.01,
                             max_candles_per_symbol=9000)
        cls.dir = decompose(root / "dir", horizon=HORIZON, barrier=0.010, budget=0.01,
                            max_candles_per_symbol=9000)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_magnitude_is_detected_when_it_is_there(self):
        self.assertEqual(self.coin["status"], "ok", self.coin)
        move = self.coin["move"]
        self.assertEqual(move["status"], "ok", move)
        self.assertGreater(move["lift_ci_low_clustered"], 1.0, "planted volatility signal went undetected")

    def test_direction_is_not_claimed_on_a_coin_flip(self):
        """The failure that would misdirect the whole project."""
        d = self.coin["direction"]
        self.assertEqual(d["status"], "ok", d)
        self.assertAlmostEqual(d["base_rate"], 0.5, delta=0.12)
        self.assertLessEqual(
            d["lift_ci_low_clustered"], 1.0,
            f"claimed directional skill where the sign is a fair coin (lift {d['lift']:.2f})",
        )

    def test_verdict_names_a_volatility_scanner(self):
        self.assertEqual(self.coin["verdict"], "magnitude_only")
        self.assertIn("volatility", format_report(self.coin).lower())

    def test_direction_is_detected_when_the_sign_is_learnable(self):
        """The complement: the tool must not be blind to real directional skill."""
        d = self.dir["direction"]
        self.assertEqual(d["status"], "ok", d)
        self.assertGreater(
            d["lift_ci_low_clustered"], 1.0,
            f"missed a directional signal leaked straight into a feature (lift {d['lift']:.2f})",
        )
        self.assertEqual(self.dir["verdict"], "both")

    def test_direction_is_measured_at_comparable_power(self):
        """Scored on the move subset, the same budget fraction would fire far
        fewer alerts and understate direction for want of observations."""
        d, m = self.coin["direction"], self.coin["move"]
        self.assertGreater(d["budget"], self.coin["budget"])
        self.assertGreater(d["alerts"], 0.5 * m["alerts"])

    def test_the_factor_arithmetic_holds(self):
        """P(up_hit) should be about P(move) x P(up | move)."""
        p = self.coin["pump"]["base_rate"]
        implied = self.coin["move"]["base_rate"] * self.coin["direction"]["base_rate"]
        self.assertAlmostEqual(p, implied, delta=0.03)

    def test_missing_features_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(decompose(Path(td))["status"], "no_features")


if __name__ == "__main__":
    unittest.main()
