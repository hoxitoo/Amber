"""One-sided labels change what up_hit/down_hit mean, and PnL must not follow.

Under the triple barrier the two flags are mutually exclusive, and several
places relied on that without saying so. Under one_sided both are 1 whenever
price touched both levels, so any `if up_hit ... elif down_hit` chain silently
books a dip-then-rally as a clean win. These tests pin the separation: the
model's target follows `label_shape`, trade accounting always follows
`first_hit`.
"""

import json
import tempfile
import unittest
from pathlib import Path

from amber.backtest.backtester import _first_touch, _label_replay
from amber.datasets.build import build_dataset
from amber.labeling.events import label_event_path, label_path
from amber.models.features import MODEL_FEATURES

DIP_THEN_RALLY = [100.0, 98.5, 101.5]  # touches -1% first, then +1%


class TestLabelPath(unittest.TestCase):
    def test_two_sided_is_unchanged(self):
        self.assertEqual(
            label_path(DIP_THEN_RALLY, up_pct=0.01, down_pct=0.01, shape="two_sided"),
            label_event_path(DIP_THEN_RALLY, up_pct=0.01, down_pct=0.01),
        )

    def test_one_sided_marks_both_barriers(self):
        one = label_path(DIP_THEN_RALLY, up_pct=0.01, down_pct=0.01, shape="one_sided")
        self.assertEqual((one["up_hit"], one["down_hit"]), (1, 1))

    def test_one_sided_keeps_first_touch_intact(self):
        """The field trade accounting depends on must not move with the shape."""
        one = label_path(DIP_THEN_RALLY, up_pct=0.01, down_pct=0.01, shape="one_sided")
        two = label_path(DIP_THEN_RALLY, up_pct=0.01, down_pct=0.01, shape="two_sided")
        self.assertEqual(one["first_hit"], -1)  # price fell first
        self.assertEqual(one["first_hit"], two["first_hit"])
        self.assertEqual(one["tte_idx"], two["tte_idx"])

    def test_unknown_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            label_path(DIP_THEN_RALLY, up_pct=0.01, down_pct=0.01, shape="sideways")


class TestBacktestIgnoresLabelShape(unittest.TestCase):
    def test_a_dip_then_rally_is_booked_as_a_loss_under_both_shapes(self):
        """The regression this whole separation exists to prevent.

        With one-sided labels this row has up_hit=1 AND down_hit=1. Reading the
        pair, the old code hit `if up == 1` first and booked a win — a stop-loss
        that would have been taken in reality, counted as profit.
        """
        rows = []
        for shape in ("two_sided", "one_sided"):
            lab = label_path(DIP_THEN_RALLY, up_pct=0.01, down_pct=0.01, shape=shape)
            rows.append({**lab, "up_pct": 0.01, "down_pct": 0.01})

        for row in rows:
            self.assertEqual(_first_touch(row), -1)
        pnls, counts = _label_replay(rows, cost=0.0009)

        self.assertEqual(counts, {"TP": 0, "SL": 2, "Timeout": 0})
        self.assertTrue(all(p < 0 for p in pnls), pnls)

    def test_first_touch_falls_back_for_rows_without_the_field(self):
        self.assertEqual(_first_touch({"up_hit": 1, "down_hit": 0}), 1)
        self.assertEqual(_first_touch({"up_hit": 0, "down_hit": 1}), -1)
        self.assertEqual(_first_touch({"up_hit": 0, "down_hit": 0}), 0)


def _write_features(root: Path, symbol: str, prices: list[float]) -> None:
    d = root / "features" / symbol
    d.mkdir(parents=True, exist_ok=True)
    with (d / "part-000.jsonl").open("w", encoding="utf-8") as fh:
        for i, price in enumerate(prices):
            row = {name: 0.0 for name in MODEL_FEATURES}
            row.update({
                "ts": 1_700_000_000_000 + i * 60_000,
                "mid_price": price,
                "obs": 200,
                "is_synthetic": False,
            })
            fh.write(json.dumps(row) + "\n")


class TestBuildDatasetShape(unittest.TestCase):
    def _build(self, root: Path, shape: str) -> list[dict]:
        # a saw-tooth that repeatedly dips below -1% then recovers above +1%
        prices = []
        for i in range(400):
            prices.extend([100.0, 98.0, 102.0, 100.0][i % 4] * (1.0 + i * 0.0001) for _ in [0])
        _write_features(root, "AAAUSDT", prices)
        out = build_dataset(
            features_root=root,
            datasets_root=root / "datasets",
            symbols=["AAAUSDT"],
            horizon_steps=15,
            up_pct=0.01,
            down_pct=0.01,
            adaptive_thresholds=False,
            min_warmup_bars=60,
            label_shape=shape,
        )
        path = root / "datasets" / out["run_id"] / "dataset.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def test_one_sided_yields_at_least_as_many_positives(self):
        with tempfile.TemporaryDirectory() as td:
            two = self._build(Path(td) / "a", "two_sided")
            one = self._build(Path(td) / "b", "one_sided")

            self.assertEqual(len(two), len(one))
            self.assertGreater(sum(r["up_hit"] for r in one), sum(r["up_hit"] for r in two))
            # rows where both barriers were touched exist only under one_sided
            self.assertGreater(sum(1 for r in one if r["up_hit"] and r["down_hit"]), 0)
            self.assertEqual(sum(1 for r in two if r["up_hit"] and r["down_hit"]), 0)

    def test_first_hit_is_identical_across_shapes(self):
        with tempfile.TemporaryDirectory() as td:
            two = self._build(Path(td) / "a", "two_sided")
            one = self._build(Path(td) / "b", "one_sided")
            self.assertEqual([r["first_hit"] for r in two], [r["first_hit"] for r in one])

    def test_shape_is_recorded_in_the_manifest(self):
        """A dataset must say which target it was labelled for."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prices = [100.0 + (i % 7) for i in range(300)]
            _write_features(root, "AAAUSDT", prices)
            out = build_dataset(
                features_root=root,
                datasets_root=root / "datasets",
                symbols=["AAAUSDT"],
                horizon_steps=15,
                up_pct=0.01,
                down_pct=0.01,
                adaptive_thresholds=False,
                min_warmup_bars=60,
                label_shape="one_sided",
            )
            manifest = json.loads((root / "datasets" / out["run_id"] / "manifest.json").read_text())
            self.assertEqual(manifest["metadata"]["label_shape"], "one_sided")

    def test_bad_shape_is_rejected_before_any_work(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                build_dataset(
                    features_root=Path(td),
                    datasets_root=Path(td) / "datasets",
                    symbols=["AAAUSDT"],
                    label_shape="diagonal",
                )


class TestConfigMatchesTheSweep(unittest.TestCase):
    def test_live_config_uses_the_arm_the_sweep_selected(self):
        """The config is the deliverable of the sweep; drift here is silent."""
        from amber.common.config import ConfigLoader

        labeling = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")["labeling"]

        self.assertFalse(labeling["adaptive_thresholds"], "adaptive barrier was the worst of 24 arms")
        self.assertEqual(labeling["up_pct"], 0.010)
        self.assertEqual(labeling["down_pct"], 0.010)
        self.assertEqual(labeling["label_shape"], "one_sided")
        self.assertEqual(labeling["horizon_steps"], 15)


if __name__ == "__main__":
    unittest.main()
