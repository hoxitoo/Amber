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
