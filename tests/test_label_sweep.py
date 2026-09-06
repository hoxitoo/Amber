"""The label sweep must measure honestly and change nothing.

Its whole purpose is to decide how the target is defined, so a bug here would
send the project down the wrong branch while looking authoritative. These tests
pin the parts that could be silently wrong: label semantics, the lag pairing,
censoring, and that a run leaves the box untouched.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from amber.backtest.label_sweep import (
    DEFAULT_BUDGET,
    RULERS,
    SHAPES,
    _episodes,
    _lagged,
    _precision_at_budget,
    build_arm_rows,
    family_z,
    format_table,
    label_one_sided,
    load_series,
    run_sweep,
)
from amber.labeling.events import label_event_path
from amber.models.features import MODEL_FEATURES


def _write_symbol(root: Path, symbol: str, rows: list[dict]) -> None:
    d = root / "features" / symbol
    d.mkdir(parents=True, exist_ok=True)
    with (d / "part-000.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _series_rows(n: int, seed: int, *, edge: bool = True, vol: float = 0.0012) -> list[dict]:
    """Candles where a flagged bar is followed by upward drift."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        r = {name: rng.gauss(0, 1) for name in MODEL_FEATURES}
        r["vol_z_20"] = 3.0 if rng.random() < 0.05 else rng.gauss(0, 1)
        r["ret_1"] = rng.gauss(0.0, vol)
        rows.append(r)
    price, ts = 100.0, 1_700_000_000_000
    for i, r in enumerate(rows):
        flagged = edge and any(rows[j]["vol_z_20"] > 2.5 for j in range(max(0, i - 3), i))
        price *= 1 + r["ret_1"] + (0.0015 if flagged else 0.0)
        r.update({"ts": ts + i * 60_000, "mid_price": price, "obs": 200, "is_synthetic": False})
    return rows


class TestLabelSemantics(unittest.TestCase):
    def test_one_sided_ignores_the_dip_that_two_sided_stops_out_on(self):
        # dips 1% first, then rallies 2%: a stop-loss label calls this a loss,
        # an alert saying "a pump is coming" was right.
        path = [100.0, 99.0, 102.0]
        two = label_event_path(path, up_pct=0.005, down_pct=0.005)
        one = label_one_sided(path, up_pct=0.005, down_pct=0.005)

        self.assertEqual((two["up_hit"], two["down_hit"]), (0, 1))
        self.assertEqual((one["up_hit"], one["down_hit"]), (1, 1))

    def test_one_sided_base_rate_is_never_below_two_sided(self):
        """Mechanically true, and the sweep's ranking depends on it holding."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_symbol(root, "AAAUSDT", _series_rows(1200, seed=3))
            series = load_series(root, max_candles_per_symbol=900)

            two = build_arm_rows(series, horizon=15, ruler="fast_vol", shape="two_sided")
            one = build_arm_rows(series, horizon=15, ruler="fast_vol", shape="one_sided")

            self.assertEqual(len(two), len(one))
            self.assertGreaterEqual(
                sum(r["up_hit"] for r in one), sum(r["up_hit"] for r in two)
            )

    def test_censored_rows_are_dropped_not_labelled_negative(self):
        """Labelling an unfinished window "no event" would bias the base rate."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_symbol(root, "AAAUSDT", _series_rows(800, seed=5))
            series = load_series(root, max_candles_per_symbol=600)
            labelled = len(series["AAAUSDT"].clean_idx)

            short = build_arm_rows(series, horizon=15, ruler="fast_vol", shape="two_sided")
            long = build_arm_rows(series, horizon=60, ruler="fast_vol", shape="two_sided")

            self.assertEqual(labelled - len(short), 15)
            self.assertEqual(labelled - len(long), 60)


class TestRulers(unittest.TestCase):
    def test_fixed_rulers_ask_the_same_move_of_every_row(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_symbol(root, "AAAUSDT", _series_rows(1000, seed=8))
            series = load_series(root, max_candles_per_symbol=800)

            rows = build_arm_rows(series, horizon=15, ruler="fixed_070", shape="two_sided")
            self.assertEqual({round(r["up_pct"], 6) for r in rows}, {0.007})

    def test_slow_ruler_differs_from_the_fast_one_it_replaces(self):
        """If both rulers produced the same barrier the arm would be a no-op."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_symbol(root, "AAAUSDT", _series_rows(2000, seed=9, vol=0.006))
            series = load_series(root, max_candles_per_symbol=1200)

            fast = build_arm_rows(series, horizon=15, ruler="fast_vol", shape="two_sided")
            slow = build_arm_rows(series, horizon=15, ruler="slow_vol", shape="two_sided")

            differing = sum(1 for a, b in zip(fast, slow) if abs(a["up_pct"] - b["up_pct"]) > 1e-9)
            self.assertGreater(differing, len(fast) * 0.2, "slow ruler tracked the fast one")

    def test_floor_clamp_is_reported(self):
        """At low volatility a "scaled" barrier is really the floor, and the
        report has to say so or the ruler comparison is meaningless."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_symbol(root, "AAAUSDT", _series_rows(1200, seed=10, vol=0.0002))
            series = load_series(root, max_candles_per_symbol=900)

            rows = build_arm_rows(series, horizon=15, ruler="fast_vol", shape="two_sided", floor=0.005)
            self.assertTrue(all(r["_clamp"] == "floor" for r in rows))
            self.assertEqual({round(r["up_pct"], 6) for r in rows}, {0.005})


class TestLagPairing(unittest.TestCase):
    def test_lag_pairs_the_score_with_the_next_bars_label(self):
        base = 1_700_000_000_000
        rows = [
            {"symbol": "A", "_i": 0, "up_hit": 0, "ts": base},
            {"symbol": "A", "_i": 1, "up_hit": 1, "ts": base + 60_000},
            {"symbol": "A", "_i": 2, "up_hit": 0, "ts": base + 120_000},
        ]
        scores = [0.9, 0.8, 0.7]

        s0, y0, _ = _lagged(rows, scores, 0, "up_hit")
        s1, y1, _ = _lagged(rows, scores, 1, "up_hit")

        self.assertEqual((s0, y0), ([0.9, 0.8, 0.7], [0, 1, 0]))
        # last row has no successor in this segment -> dropped, not guessed
        self.assertEqual((s1, y1), ([0.9, 0.8], [1, 0]))

    def test_lag_does_not_cross_symbols(self):
        rows = [
            {"symbol": "A", "_i": 5, "up_hit": 0, "ts": 1_700_000_000_000},
            {"symbol": "B", "_i": 6, "up_hit": 1, "ts": 1_700_000_060_000},
        ]
        s1, y1, _ = _lagged(rows, [0.9, 0.1], 1, "up_hit")
        self.assertEqual((s1, y1), ([], []))


class TestPrecisionAtBudget(unittest.TestCase):
    def test_precision_is_measured_on_the_top_scored_rows(self):
        scores = [0.1, 0.9, 0.2, 0.8, 0.3]
        labels = [0, 1, 0, 1, 0]
        res = _precision_at_budget(scores, labels, [0] * len(scores), budget=0.4)

        self.assertEqual(res["alerts"], 2)
        self.assertEqual(res["precision"], 1.0)
        self.assertAlmostEqual(res["base_rate"], 0.4)
        self.assertAlmostEqual(res["lift"], 2.5)

    def test_a_useless_score_lands_at_lift_one(self):
        labels = [1 if i % 4 == 0 else 0 for i in range(400)]
        scores = [0.5] * 400  # no ordering information at all
        res = _precision_at_budget(scores, labels, [i * 60_000 * 60 for i in range(400)], budget=0.1)
        self.assertAlmostEqual(res["lift"], 1.0, delta=0.35)


class TestEpisodeClustering(unittest.TestCase):
    """Alerts are not independent observations, and the interval must know it."""

    def test_simultaneous_alerts_across_symbols_are_one_episode(self):
        # a market-wide lurch: 40 alerts, same minute, 40 different coins
        ts = [1_700_000_000_000] * 40
        self.assertEqual(_episodes(ts, horizon=15), 1)

    def test_alerts_inside_one_horizon_are_one_episode(self):
        base = 1_700_000_000_000
        ts = [base + i * 60_000 for i in range(10)]  # 10 consecutive minutes
        self.assertEqual(_episodes(ts, horizon=15), 1)

    def test_well_separated_alerts_count_separately(self):
        base = 1_700_000_000_000
        ts = [base + i * 60_000 * 60 for i in range(5)]  # an hour apart
        self.assertEqual(_episodes(ts, horizon=15), 5)

    def test_clustered_bound_is_weaker_than_the_naive_one(self):
        """The whole point: 100 alerts from one lurch are not 100 observations."""
        base_ts = 1_700_000_000_000
        scores = [1.0] * 100 + [0.0] * 900
        labels = [1] * 80 + [0] * 20 + [0] * 900
        # every alert lands in the same minute -> one episode
        timestamps = [base_ts] * 100 + [base_ts + i * 60_000 for i in range(900)]

        res = _precision_at_budget(scores, labels, timestamps, budget=0.1, horizon=15)

        self.assertEqual(res["episodes"], 1)
        self.assertAlmostEqual(res["precision"], 0.8)
        self.assertLess(
            res["lift_ci_low_clustered"],
            res["lift_ci_low"],
            "clustering did not weaken the bound",
        )

    def test_spread_out_alerts_keep_their_evidence(self):
        """The correction must not punish genuinely independent alerts."""
        base_ts = 1_700_000_000_000
        scores = [1.0] * 100 + [0.0] * 900
        labels = [1] * 80 + [0] * 20 + [0] * 900
        timestamps = [base_ts + i * 60_000 * 60 for i in range(1000)]  # hours apart

        res = _precision_at_budget(scores, labels, timestamps, budget=0.1, horizon=15)

        self.assertEqual(res["episodes"], 100)
        self.assertAlmostEqual(res["lift_ci_low_clustered"], res["lift_ci_low"], places=6)


class TestSweepEndToEnd(unittest.TestCase):
    def test_sweep_finds_a_planted_edge(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i, sym in enumerate(("AAAUSDT", "BBBUSDT", "CCCUSDT")):
                _write_symbol(root, sym, _series_rows(3200, seed=20 + i))

            rep = run_sweep(
                root,
                horizons=(15,),
                rulers=("fast_vol", "fixed_070"),
                shapes=("two_sided",),
                max_candles_per_symbol=2400,
                budget=0.01,
            )

            self.assertEqual(rep["status"], "ok")
            self.assertEqual(rep["symbols"], 3)
            ok = [a for a in rep["arms"] if a["status"] == "ok"]
            self.assertTrue(ok, rep["arms"])
            best_lift = max((a["test"]["lag0"]["lift"] or 0) for a in ok)
            self.assertGreater(best_lift, 1.1, "planted edge went undetected")

    def test_sweep_writes_nothing(self):
        """It is meant to run against a live box mid-flight."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_symbol(root, "AAAUSDT", _series_rows(2200, seed=31))
            _write_symbol(root, "BBBUSDT", _series_rows(2200, seed=32))
            before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))

            run_sweep(
                root,
                horizons=(15,),
                rulers=("fast_vol",),
                shapes=("two_sided",),
                max_candles_per_symbol=1800,
                budget=DEFAULT_BUDGET,
            )

            after = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
            self.assertEqual(before, after)

    def test_random_walk_yields_no_edge(self):
        """The null test, and the reason the verdict uses a confidence bound.

        On a pure random walk the first version of this sweep reported a top arm
        at lift 1.78 — precision measured on ~20 alerts. Ranking on the point
        estimate would have sent the project off to redefine its target on the
        strength of noise. Nothing here may claim an edge where none exists.

        The arm grid is the full one, deliberately. An earlier version of this
        test used 8 arms and passed, while the real 24-arm run on the same kind
        of random data returned `edge_survives_lag`: with enough arms something
        crosses a per-arm bound by luck, which is what `family_z` corrects.
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i, sym in enumerate(("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT")):
                _write_symbol(root, sym, _series_rows(3000, seed=40 + i, edge=False))

            rep = run_sweep(
                root,
                horizons=(15, 30, 60),
                rulers=RULERS,
                shapes=SHAPES,
                max_candles_per_symbol=2400,
                budget=0.01,
            )

            self.assertEqual(rep["n_arms"], 24)
            self.assertNotEqual(rep["verdict"], "edge_survives_lag")
            for arm in (a for a in rep["arms"] if a["status"] == "ok"):
                self.assertLessEqual(
                    arm["test"]["lag1"]["lift_ci_low"] or 0.0,
                    1.0,
                    f"claimed a significant edge on a random walk: {arm['ruler']}/{arm['shape']}",
                )

    def test_more_arms_demand_a_stronger_result(self):
        """The correction has to actually tighten as the family grows."""
        self.assertGreater(family_z(24), family_z(4))
        self.assertAlmostEqual(family_z(1), 1.645, places=2)  # one-sided 95%

    def test_underpowered_runs_are_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_symbol(root, "AAAUSDT", _series_rows(3000, seed=50))
            _write_symbol(root, "BBBUSDT", _series_rows(3000, seed=51))

            rep = run_sweep(
                root,
                horizons=(15,),
                rulers=("fast_vol",),
                shapes=("two_sided",),
                max_candles_per_symbol=2400,
                budget=0.002,  # a handful of alerts on a short test segment
            )

            self.assertTrue(rep["underpowered"])
            self.assertLess(rep["test_alerts_per_arm"], 60)
            self.assertIn("WARNING", format_table(rep))

    def test_missing_features_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as td:
            rep = run_sweep(Path(td), horizons=(15,))
            self.assertEqual(rep["status"], "no_features")


if __name__ == "__main__":
    unittest.main()
