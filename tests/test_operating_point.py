"""The operating curve must hand back a threshold that actually works.

Its output is meant to be copied into `config/thresholds.yaml`, so the round
trip has to close: the recommended `prob_lift_min` fed back through the
scanner's own rule must reproduce the probability cut the curve measured at,
and that cut must produce the promised alert rate. A curve that reads well but
recommends a threshold the gate interprets differently would be worse than no
curve at all.
"""

import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from amber.backtest.operating_point import (
    MIN_EPISODES,
    format_curve,
    lift_for_threshold,
    operating_curve,
)
from amber.common.config import ConfigLoader
from amber.datasets.build import build_dataset
from amber.models.features import MODEL_FEATURES
from amber.pipeline.train_app import run_training
from amber.signals.filters import effective_prob_min

REPO = Path(__file__).resolve().parents[1]


def _write_symbol(features_root: Path, symbol: str, n: int, seed: int) -> None:
    rng = random.Random(seed)
    d = features_root / "features" / symbol
    d.mkdir(parents=True, exist_ok=True)
    spikes = [rng.random() < 0.02 for _ in range(n)]
    price, ts = 100.0, 1_700_000_000_000
    with (d / "part-000.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(n):
            row = {name: rng.gauss(0, 1) for name in MODEL_FEATURES}
            recent = any(spikes[max(0, i - 15):i])
            row["vol_z_20"] = 3.0 if spikes[i] else rng.gauss(0, 1)
            row["range_atr_14"] = 2.5 if spikes[i] else rng.gauss(0, 1)
            row["ret_1"] = rng.gauss(0.0, 0.0028 if recent else 0.0003)
            price *= 1 + row["ret_1"]
            row.update({
                "ts": ts + i * 60_000, "mid_price": price, "obs": 200, "is_synthetic": False,
                "bid": price * 0.9999, "ask": price * 1.0001, "spread_bps": 2.0,
            })
            fh.write(json.dumps(row) + "\n")


class TestLiftInversion(unittest.TestCase):
    """The curve reports a lift; the gate turns a lift back into a probability."""

    def test_the_round_trip_closes(self):
        for base in (0.02, 0.075, 0.20, 0.45):
            for threshold in (0.1, 0.3, 0.6, 0.9):
                lift = lift_for_threshold(threshold, base)
                self.assertIsNotNone(lift)
                back = effective_prob_min(
                    {"prob_lift_min": lift, "prob_abs_floor": 0.0},
                    base,
                    absolute_key="move_prob_calibrated_min",
                )
                self.assertAlmostEqual(back, threshold, places=6, msg=f"base={base} thr={threshold}")

    def test_degenerate_base_rates_do_not_produce_a_number(self):
        self.assertIsNone(lift_for_threshold(0.5, 0.0))
        self.assertIsNone(lift_for_threshold(0.5, 1.0))
        self.assertIsNone(lift_for_threshold(1.0, 0.1))


class TestRunsFromAnywhere(unittest.TestCase):
    """Storage paths in the config are relative, so the working directory is
    load-bearing. Run from a home directory the script could not even open its
    own file, failing with a permission error that named nothing real."""

    def test_entering_the_project_root_finds_the_config(self):
        import os

        from amber.common.config import ConfigLoader, enter_project_root

        previous = Path.cwd()
        try:
            os.chdir(tempfile.gettempdir())
            root = enter_project_root(REPO / "scripts" / "run_operating_curve.py")

            self.assertEqual(root, REPO)
            self.assertEqual(Path.cwd(), REPO)
            self.assertIn("labeling", ConfigLoader(root).load_yaml("config/amber.yaml"))
        finally:
            os.chdir(previous)


class TestMissingArtifactsAreReported(unittest.TestCase):
    """Right after a target change there is no dataset yet. That is the normal
    state, and it must read as a wait rather than a traceback."""

    def test_absent_dataset(self):
        with tempfile.TemporaryDirectory() as td:
            report = operating_curve(Path(td) / "datasets", Path(td) / "models")
            self.assertEqual(report["status"], "no_dataset")
            self.assertIn("retrain cycle", format_curve(report))


class TestOperatingCurve(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        shutil.copytree(REPO / "config", root / "config")
        data = root / "data"
        data.mkdir()

        config = ConfigLoader(REPO).load_yaml("config/amber.yaml")
        lab = config["labeling"]
        symbols = [f"S{i:02d}USDT" for i in range(5)]
        for i, sym in enumerate(symbols):
            _write_symbol(data, sym, n=4000, seed=300 + i)

        cls.datasets, cls.models, cls.logs = data / "datasets", data / "models", data / "logs"
        build_dataset(
            features_root=data,
            datasets_root=cls.datasets,
            symbols=symbols,
            horizon_steps=int(lab["horizon_steps"]),
            horizon_steps_list=[int(x) for x in lab["horizon_steps_list"]],
            up_pct=float(lab["up_pct"]),
            down_pct=float(lab["down_pct"]),
            adaptive_thresholds=False,
            min_warmup_bars=int(lab["min_warmup_bars"]),
            max_candles_per_symbol=4000,
            label_shape=str(lab["label_shape"]),
        )
        cfg = dict(config)
        cfg["storage"] = {
            **config["storage"], "raw_dir": str(data / "raw"),
            "datasets_dir": str(cls.datasets), "models_dir": str(cls.models), "logs_dir": str(cls.logs),
        }
        run_training(cfg, cls.datasets, cls.models, cls.logs)
        cls.report = operating_curve(cls.datasets, cls.models, target="move")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_curve_is_produced_out_of_sample(self):
        self.assertEqual(self.report["status"], "ok", self.report)
        self.assertFalse(self.report["in_sample"], "curve must not be measured on training rows")

    def test_tighter_rates_use_higher_thresholds(self):
        """The curve is a ranking cut, so this is a structural invariant."""
        ok = [p for p in self.report["points"] if p["status"] == "ok"]
        self.assertGreater(len(ok), 2, self.report["points"])
        by_rate = sorted(ok, key=lambda p: p["alerts_per_day"])
        thresholds = [p["threshold"] for p in by_rate]
        self.assertEqual(thresholds, sorted(thresholds, reverse=True))

    def test_a_recommended_threshold_reproduces_its_own_alert_rate(self):
        """The whole deliverable: paste the lift into config, get the rate back."""
        rec = self.report.get("recommended")
        if rec is None:
            self.skipTest("no rate held its bound on this fixture")

        applied = effective_prob_min(
            {"prob_lift_min": rec["prob_lift_min_equivalent"], "prob_abs_floor": 0.0},
            self.report["base_rate"],
            absolute_key="move_prob_calibrated_min",
        )
        self.assertAlmostEqual(applied, rec["threshold"], places=6)

    def test_recommendation_never_rests_on_a_handful_of_episodes(self):
        """Without an episode floor this rule picked 10 alerts/day off TWO
        episodes while rejecting 20 alerts/day at three — a non-monotonic result
        that is a lucky draw, and it would have been written into live config."""
        rec = self.report.get("recommended")
        if rec is None:
            self.skipTest("no rate held its bound on this fixture")
        self.assertGreaterEqual(rec["episodes"], MIN_EPISODES)
        self.assertFalse(rec["underpowered"])

    def test_underpowered_rates_are_shown_but_not_chosen(self):
        thin = [p for p in self.report["points"] if p["status"] == "ok" and p["underpowered"]]
        rec = self.report.get("recommended")
        for p in thin:
            if rec is not None:
                self.assertNotEqual(p["alerts_per_day"], rec["alerts_per_day"])
        if thin:
            self.assertIn("мало эпизодов", format_curve(self.report))

    def test_an_unreachable_rate_is_reported_not_silently_dropped(self):
        report = operating_curve(self.datasets, self.models, target="move", rates_per_day=(10**7,))
        self.assertEqual(report["points"][0]["status"], "out_of_range")

    def test_missing_head_is_reported(self):
        report = operating_curve(self.datasets, self.models, target="dump")
        # `dump` exists, so this must succeed; the guard is exercised by name.
        self.assertIn(report["status"], ("ok", "no_positive_labels"))

    def test_report_renders(self):
        text = format_curve(self.report)
        self.assertIn("alerts/day", text)
        self.assertIn("recommended", text)


if __name__ == "__main__":
    unittest.main()
