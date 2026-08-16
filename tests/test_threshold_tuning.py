"""The sweep must select and validate on different data, and never auto-apply."""

import tempfile
import unittest
from pathlib import Path

import yaml

from amber.backtest.tuning import load_sweep, save_sweep
from amber.common.config import ConfigLoader
from amber.dashboard.control import apply_thresholds


class TestSweepPersistence(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            logs = Path(td)
            self.assertIsNone(load_sweep(logs))
            save_sweep(logs, {"status": "ok", "verdict": "holds", "grid": []})
            self.assertEqual(load_sweep(logs)["verdict"], "holds")

    def test_corrupt_file_is_not_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            logs = Path(td)
            (logs / "threshold_sweep.json").write_text("{not json", encoding="utf-8")
            self.assertIsNone(load_sweep(logs))

    def test_no_temp_file_left_behind(self):
        with tempfile.TemporaryDirectory() as td:
            logs = Path(td)
            save_sweep(logs, {"status": "ok"})
            self.assertEqual([p.name for p in logs.iterdir()], ["threshold_sweep.json"])


class TestApplyThresholds(unittest.TestCase):
    def _project(self, td: str) -> Path:
        root = Path(td)
        (root / "config").mkdir()
        (root / "config" / "thresholds.yaml").write_text(
            "thresholds:\n  prob_lift_min: 2.0\n  prob_abs_floor: 0.12\n"
            "  directional_score_min: 0.05\n  cooldown_sec: 90\n",
            encoding="utf-8",
        )
        return root

    def test_writes_local_override_leaving_tracked_file_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._project(td)
            tracked = root / "config" / "thresholds.yaml"
            before = tracked.read_text(encoding="utf-8")

            apply_thresholds(root, 2.5, 0.10)

            self.assertEqual(tracked.read_text(encoding="utf-8"), before)
            local = yaml.safe_load((root / "config" / "thresholds.local.yaml").read_text(encoding="utf-8"))
            self.assertEqual(local["thresholds"]["prob_lift_min"], 2.5)

    def test_override_is_merged_on_load_and_keeps_other_keys(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._project(td)
            apply_thresholds(root, 3.0, 0.0)
            merged = ConfigLoader(root).load_yaml("config/thresholds.yaml")["thresholds"]
            self.assertEqual(merged["prob_lift_min"], 3.0)
            self.assertEqual(merged["directional_score_min"], 0.0)
            self.assertEqual(merged["cooldown_sec"], 90)  # untouched keys survive
            self.assertEqual(merged["prob_abs_floor"], 0.12)

    def test_applying_twice_updates_rather_than_duplicates(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._project(td)
            apply_thresholds(root, 2.5, 0.05)
            apply_thresholds(root, 1.5, 0.10)
            merged = ConfigLoader(root).load_yaml("config/thresholds.yaml")["thresholds"]
            self.assertEqual(merged["prob_lift_min"], 1.5)
            self.assertEqual(merged["directional_score_min"], 0.10)


class TestVerdictLogic(unittest.TestCase):
    """A point that wins on selection but fails validation must be called out,
    not presented as an edge."""

    def test_pipeline_never_applies_thresholds_by_itself(self):
        src = (Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline_loop.py").read_text(encoding="utf-8")
        self.assertNotIn("apply_thresholds", src)
        self.assertIn("save_sweep", src)


if __name__ == "__main__":
    unittest.main()


class TestSweepMatchesTheLiveGate(unittest.TestCase):
    """The sweep and the promotion-gate backtest must apply identical rules.

    They used to hold separate copies, and the sweep's had lost the spread
    filter — so it validated a gate the scanner does not run and could recommend
    a threshold that behaves differently in production.
    """

    def _rows(self, n=400, spread=1.0):
        return [
            {
                "symbol": "BTCUSDT", "ts": 1_700_000_000_000 + i * 60_000,
                "up_hit": i % 3 == 0, "down_hit": i % 5 == 0,
                "up_pct": 0.008, "down_pct": 0.008, "horizon_steps": 15,
                "spread_bps": spread,
            }
            for i in range(n)
        ]

    def test_spread_filter_is_applied_by_the_sweep(self):
        from amber.backtest.tuning import _replay

        rows = self._rows(spread=99.0)  # every row above any sane spread cap
        probs = [(0.9, 0.1)] * len(rows)
        wide = _replay(rows, probs, 0.5, 0.5, 0.0, spread_max=1000.0, cost=0.0009)
        tight = _replay(rows, probs, 0.5, 0.5, 0.0, spread_max=30.0, cost=0.0009)
        self.assertGreater(wide["trades"], 0)
        self.assertEqual(tight["trades"], 0, "sweep ignored the spread filter the scanner enforces")

    def test_sweep_and_backtest_book_identical_trades(self):
        from amber.backtest.backtester import replay_with_probs
        from amber.backtest.tuning import _replay

        rows = self._rows(spread=5.0)
        probs = [(0.9, 0.1) if i % 4 else (0.1, 0.9) for i in range(len(rows))]
        pnl, counts, _ = replay_with_probs(
            rows, probs, up_min=0.5, down_min=0.5, dir_min=0.0, spread_max=30.0, cost=0.0009
        )
        swept = _replay(rows, probs, 0.5, 0.5, 0.0, spread_max=30.0, cost=0.0009)
        self.assertEqual(swept["trades"], len(pnl))
        self.assertAlmostEqual(swept["expectancy"], sum(pnl) / len(pnl), places=12)
        self.assertAlmostEqual(swept["win_rate"], counts["TP"] / (counts["TP"] + counts["SL"]), places=12)

    def test_backtester_exposes_one_shared_replay(self):
        """Guard against a second copy reappearing."""
        src = (Path(__file__).resolve().parents[1] / "amber" / "backtest" / "tuning.py").read_text(encoding="utf-8")
        self.assertIn("replay_with_probs", src)
        self.assertNotIn("counts = {\"TP\"", src)
