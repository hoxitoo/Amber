"""The training window and split decide how many independent episodes the
verdicts are drawn from, and the window alone decides the RAM a retrain needs.

Every symbol shares one time axis, so the test segment is a stretch of wall
clock, not a count of rows: adding symbols adds rows but not episodes. At a
48h window with a 15% test share that stretch was 7.2h and produced 8 episodes
live (2026-09-29), below MIN_EPISODES = 10, so no verdict could be issued.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

from amber.backtest import label_sweep
from amber.backtest.baselines import MIN_EPISODES
from amber.common.config import ConfigLoader

REPO = Path(__file__).resolve().parents[1]

# Measured, not derived: 8 episodes in a 7.2h test segment on the live box.
# The rate depends on the market, so this is a planning figure, and the test
# keeps a margin above it rather than aiming at exactly MIN_EPISODES.
OBSERVED_EPISODES_PER_HOUR = 8 / 7.2

# The largest dataset this box has already trained on without trouble:
# 2880 candles × 27 symbols × 3 horizons, before h=60 was dropped.
PROVEN_ROWS = 2880 * 27 * 3


def _cfg() -> dict:
    return ConfigLoader(REPO).load_yaml("config/amber.yaml")


class TestEpisodeBudget(unittest.TestCase):
    def test_test_segment_is_long_enough_for_a_verdict(self):
        cfg = _cfg()
        window_h = int(cfg["labeling"]["max_candles_per_symbol"]) / 60
        split = cfg["model"]["split"]
        test_share = 1.0 - float(split["train_frac"]) - float(split["calib_frac"])
        expected = window_h * test_share * OBSERVED_EPISODES_PER_HOUR
        self.assertGreaterEqual(
            expected, 1.5 * MIN_EPISODES,
            f"test segment {window_h * test_share:.1f}h ≈ {expected:.0f} episodes; "
            f"verdicts need {MIN_EPISODES} with room to spare",
        )

    def test_training_still_gets_most_of_the_window(self):
        split = _cfg()["model"]["split"]
        self.assertGreaterEqual(float(split["train_frac"]), 0.6)

    def test_dataset_stays_within_what_has_already_run(self):
        cfg = _cfg()
        rows = (
            len(cfg["exchange"]["bybit"]["symbols"])
            * int(cfg["labeling"]["max_candles_per_symbol"])
            * len(cfg["labeling"]["horizon_steps_list"])
        )
        self.assertLessEqual(rows, PROVEN_ROWS, "dataset larger than any the 3.9 GB box has trained on")


class TestAnalysisUsesTheLiveSplit(unittest.TestCase):
    """The label sweep and decomposition score the test segment too. With
    their own hard-coded 70/15/15 they would report on a different stretch of
    time than the model and the baseline check do."""

    def test_sweep_forwards_the_split_to_every_arm(self):
        seen = []

        def fake_eval(rows, **kw):
            seen.append((kw["train_frac"], kw["calib_frac"]))
            return {}

        with mock.patch.object(label_sweep, "load_series", return_value={"X": []}), \
             mock.patch.object(label_sweep, "build_arm_rows", return_value=[]), \
             mock.patch.object(label_sweep, "evaluate_arm", side_effect=fake_eval):
            label_sweep.run_sweep(Path("."), horizons=[15], rulers=["fixed_100"], shapes=["one_sided"],
                                  train_frac=0.6, calib_frac=0.15)
        self.assertEqual(seen, [(0.6, 0.15)])

    def _run_script(self, name: str, target: str) -> dict:
        spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        captured = {}

        def fake(*a, **kw):
            captured.update(kw)
            return {"status": "no_features"}

        with mock.patch.object(module, target, side_effect=fake), \
             mock.patch.object(module, "format_table", return_value="", create=True), \
             mock.patch.object(module, "format_report", return_value="", create=True), \
             mock.patch.object(sys, "argv", [name]):
            try:
                module.main()
            except SystemExit:
                pass
        return captured

    def test_scripts_read_the_split_from_config(self):
        split = _cfg()["model"]["split"]
        for name, target in (("run_label_sweep", "run_sweep"), ("run_label_decomposition", "decompose")):
            with self.subTest(script=name):
                kw = self._run_script(name, target)
                self.assertEqual(kw.get("train_frac"), float(split["train_frac"]))
                self.assertEqual(kw.get("calib_frac"), float(split["calib_frac"]))


if __name__ == "__main__":
    unittest.main()
