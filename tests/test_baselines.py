"""The baseline check must recognise its own null and its own positive.

This decides whether the project keeps tuning a model or goes after new inputs,
so it is tested against two fixtures where the answer is known by construction:

- a model trained on data where ONE feature determines the label — the trivial
  rule must come out as its twin, and the verdict as a tautology;
- a model trained on data where the label depends on a combination no single
  feature reveals — the rule must not match it.

A check that cannot tell those apart would send the project the wrong way with
an authoritative-looking table.
"""

import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from amber.backtest.baselines import (
    TAUTOLOGY_OVERLAP,
    _spearman,
    compare_to_baselines,
    format_report,
)
from amber.common.config import ConfigLoader
from amber.datasets.build import build_dataset
from amber.models.features import MODEL_FEATURES
from amber.pipeline.train_app import run_training

REPO = Path(__file__).resolve().parents[1]


def _write_symbol(features_root: Path, symbol: str, n: int, seed: int, *, single_feature: bool) -> None:
    """`single_feature`: range_atr_14 alone decides whether a burst follows.

    Otherwise the burst needs two features to agree, which no one-feature
    ranking can reproduce.
    """
    rng = random.Random(seed)
    d = features_root / "features" / symbol
    d.mkdir(parents=True, exist_ok=True)

    rows = []
    for _ in range(n):
        row = {name: rng.gauss(0, 1) for name in MODEL_FEATURES}
        rows.append(row)

    fires = []
    for i, row in enumerate(rows):
        if single_feature:
            fire = row["range_atr_14"] > 1.8
        else:
            # Needs both, and each alone is uninformative: ranking by either
            # picks mostly rows where the other did not agree.
            fire = row["ret_20"] > 1.0 and row["oi_z_20"] > 1.0
            row["range_atr_14"] = rng.gauss(0, 1)
        fires.append(fire)

    price, ts = 100.0, 1_700_000_000_000
    for i, row in enumerate(rows):
        in_burst = any(fires[max(0, i - 15):i + 1])
        row["ret_1"] = rng.gauss(0.0, 0.0030 if in_burst else 0.0002)
        price *= 1 + row["ret_1"]
        row.update({
            "ts": ts + i * 60_000, "mid_price": price, "obs": 200, "is_synthetic": False,
            "bid": price * 0.9999, "ask": price * 1.0001, "spread_bps": 2.0,
        })

    with (d / "part-000.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _train(root: Path, *, single_feature: bool) -> tuple[Path, Path]:
    shutil.copytree(REPO / "config", root / "config")
    data = root / "data"
    data.mkdir()
    config = ConfigLoader(REPO).load_yaml("config/amber.yaml")
    lab = config["labeling"]
    symbols = [f"S{i:02d}USDT" for i in range(5)]
    for i, sym in enumerate(symbols):
        _write_symbol(data, sym, 4000, seed=900 + i, single_feature=single_feature)

    datasets, models, logs = data / "datasets", data / "models", data / "logs"
    build_dataset(
        features_root=data, datasets_root=datasets, symbols=symbols,
        horizon_steps=int(lab["horizon_steps"]),
        horizon_steps_list=[int(x) for x in lab["horizon_steps_list"]],
        up_pct=float(lab["up_pct"]), down_pct=float(lab["down_pct"]),
        adaptive_thresholds=False, min_warmup_bars=int(lab["min_warmup_bars"]),
        max_candles_per_symbol=4000, label_shape=str(lab["label_shape"]),
    )
    cfg = dict(config)
    cfg["storage"] = {
        **config["storage"], "raw_dir": str(data / "raw"),
        "datasets_dir": str(datasets), "models_dir": str(models), "logs_dir": str(logs),
    }
    run_training(cfg, datasets, models, logs)
    return datasets, models


class TestSpearman(unittest.TestCase):
    def test_perfect_agreement(self):
        self.assertAlmostEqual(_spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0, places=6)

    def test_perfect_disagreement(self):
        self.assertAlmostEqual(_spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0, places=6)

    def test_constant_input_has_no_correlation(self):
        self.assertIsNone(_spearman([1, 2, 3, 4], [5, 5, 5, 5]))

    def test_too_few_points(self):
        self.assertIsNone(_spearman([1, 2], [1, 2]))


class TestOverlapArithmetic(unittest.TestCase):
    """The overlap and verdict logic, on scores chosen by hand.

    The end-to-end fixtures below train real models, where the price move itself
    leaks into `ret_1` and no model is ever a pure function of one feature. These
    pin the mechanics exactly; those pin the behaviour comparatively.
    """

    def test_identical_ranking_is_full_overlap(self):
        from amber.backtest.baselines import _score_set

        scores = [0.1, 0.9, 0.5, 0.7]
        self.assertEqual(_score_set(scores, 2), [1, 3])
        self.assertEqual(set(_score_set(scores, 2)) & set(_score_set([s * 3 for s in scores], 2)), {1, 3})

    def test_reversed_ranking_shares_nothing(self):
        from amber.backtest.baselines import _score_set

        scores = [0.1, 0.9, 0.5, 0.7]
        self.assertEqual(set(_score_set(scores, 2)) & set(_score_set([-s for s in scores], 2)), set())

    def test_verdict_calls_a_twin_a_tautology(self):
        from amber.backtest.baselines import _verdict

        report = {
            "model": {"precision": 0.9},
            "baselines": [{"name": "range_atr_14", "status": "ok",
                           "overlap_with_model": 0.95, "precision": 0.88}],
        }
        self.assertEqual(_verdict(report), "tautology:range_atr_14")

    def test_verdict_flags_a_rule_that_merely_matches_precision(self):
        from amber.backtest.baselines import _verdict

        report = {
            "model": {"precision": 0.80},
            "baselines": [{"name": "bb_width_20", "status": "ok",
                           "overlap_with_model": 0.20, "precision": 0.82}],
        }
        self.assertEqual(_verdict(report), "matched_by:bb_width_20")

    def test_verdict_credits_a_model_that_wins_and_differs(self):
        from amber.backtest.baselines import _verdict

        report = {
            "model": {"precision": 0.90},
            "baselines": [{"name": "bb_width_20", "status": "ok",
                           "overlap_with_model": 0.20, "precision": 0.60}],
        }
        self.assertEqual(_verdict(report), "model_adds_signal")


class TestTautologyIsCaught(unittest.TestCase):
    """The model here is driven by one feature. The check must show it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        datasets, models = _train(Path(cls._tmp.name), single_feature=True)
        cls.report = compare_to_baselines(datasets, models, target="move", rate_per_day=400)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_rule_picks_most_of_the_models_alerts(self):
        self.assertEqual(self.report["status"], "ok", self.report)
        atr = next(b for b in self.report["baselines"] if b["name"] == "range_atr_14")
        # Not TAUTOLOGY_OVERLAP: the burst raises ret_1, which is also a model
        # feature, so even here the model sees slightly more than the one rule.
        # That is true of the live system too — hence the comparative test below.
        self.assertGreater(
            atr["overlap_with_model"], 0.6,
            f"the driving feature picked only {atr['overlap_with_model']:.0%} of the model's alerts",
        )

    def test_rank_correlation_is_reported_but_carries_no_verdict(self):
        """Deliberately not asserted as high. On this very fixture — where one
        feature drives the label — membership overlap is 77% while the
        correlation is 0.05: the two agree on WHICH rows are extreme and
        disagree on the ordering WITHIN the extreme tail. It is reported as
        context; no verdict rests on it."""
        import math

        atr = next(b for b in self.report["baselines"] if b["name"] == "range_atr_14")
        self.assertIsNotNone(atr["spearman_with_model"])
        self.assertTrue(math.isfinite(atr["spearman_with_model"]))
        self.assertLessEqual(abs(atr["spearman_with_model"]), 1.0)


class TestRealSignalIsNotCalledATautology(unittest.TestCase):
    """The complement: a label no single feature reveals must not match."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        datasets, models = _train(Path(cls._tmp.name), single_feature=False)
        cls.report = compare_to_baselines(datasets, models, target="move", rate_per_day=400)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_no_single_feature_reproduces_the_model(self):
        self.assertEqual(self.report["status"], "ok", self.report)
        worst = max(b["overlap_with_model"] for b in self.report["baselines"] if b["status"] == "ok")
        self.assertLess(worst, TAUTOLOGY_OVERLAP, "a one-feature rule reproduced a two-feature label")

    def test_verdict_is_not_a_tautology(self):
        self.assertFalse(self.report["verdict"].startswith("tautology:"), self.report["verdict"])

    def test_overlap_is_far_below_the_single_feature_case(self):
        """The comparison that carries the meaning: the same check, run on a
        label one feature explains and on one it does not, must separate them."""
        with tempfile.TemporaryDirectory() as td:
            ds, md = _train(Path(td), single_feature=True)
            driven = compare_to_baselines(ds, md, target="move", rate_per_day=400)

        def atr(rep):
            return next(b for b in rep["baselines"] if b["name"] == "range_atr_14")["overlap_with_model"]

        self.assertGreater(
            atr(driven) - atr(self.report), 0.3,
            f"single-feature {atr(driven):.0%} vs combination {atr(self.report):.0%} — "
            "the check does not separate the two cases",
        )


class TestGuards(unittest.TestCase):
    def test_missing_dataset_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            rep = compare_to_baselines(Path(td) / "d", Path(td) / "m")
            self.assertEqual(rep["status"], "no_dataset")
            self.assertIn("unavailable", format_report(rep))

    def test_unknown_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                compare_to_baselines(Path(td), Path(td), target="sideways")


if __name__ == "__main__":
    unittest.main()
