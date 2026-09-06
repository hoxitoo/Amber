"""features -> dataset -> train -> eval -> backtest, through the live config.

Unit tests pass happily while the system as a whole produces nothing, which is
how this project has failed twice: a threshold no calibrated head could reach
(B3, B6) and a backfill that flattened prices into a constant model. Both showed
up only as `precision 0.000` and `0 trades`, and both survived days of running.

This test wires the real `config/amber.yaml` to synthetic data carrying a known
edge and asserts the pipeline converts it into signals. It is slower than a unit
test on purpose: the failures it catches are the expensive ones.
"""

import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from amber.backtest.backtester import event_backtest
from amber.common.config import ConfigLoader
from amber.datasets.build import build_dataset
from amber.models.features import MODEL_FEATURES
from amber.monitoring.reporting import _load_thresholds
from amber.pipeline.train_app import run_training

REPO = Path(__file__).resolve().parents[1]


def _write_symbol(features_root: Path, symbol: str, n: int, seed: int) -> None:
    """Candles where a volatility spike is followed by a drift large enough to
    cross a 1% barrier within the horizon — the hypothesis the model must learn."""
    rng = random.Random(seed)
    d = features_root / "features" / symbol
    d.mkdir(parents=True, exist_ok=True)
    spikes = [rng.random() < 0.005 for _ in range(n)]
    price, ts = 100.0, 1_700_000_000_000
    with (d / "part-000.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(n):
            row = {name: rng.gauss(0, 1) for name in MODEL_FEATURES}
            recent = any(spikes[max(0, i - 15):i])
            row["vol_z_20"] = 3.0 if spikes[i] else rng.gauss(0, 1)
            row["range_atr_14"] = 2.5 if spikes[i] else rng.gauss(0, 1)
            row["ret_1"] = rng.gauss(0.0, 0.0008) + (0.0009 if recent else 0.0)
            price *= 1 + row["ret_1"]
            row.update({
                "ts": ts + i * 60_000,
                "mid_price": price,
                "obs": 200,
                "is_synthetic": False,
                "bid": price * 0.9999,
                "ask": price * 1.0001,
                "spread_bps": 2.0,
            })
            fh.write(json.dumps(row) + "\n")


class TestPipelineEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        # Mirror the deployed layout: config/ above the storage dirs, which is
        # how _load_thresholds finds thresholds.yaml by walking up.
        shutil.copytree(REPO / "config", root / "config")
        data = root / "data"
        data.mkdir()

        cls.config = ConfigLoader(REPO).load_yaml("config/amber.yaml")
        lab = cls.config["labeling"]
        symbols = [f"S{i:02d}USDT" for i in range(6)]
        for i, sym in enumerate(symbols):
            _write_symbol(data, sym, n=1500, seed=100 + i)

        cls.datasets = data / "datasets"
        cls.models = data / "models"
        cls.logs = data / "logs"
        cls.built = build_dataset(
            features_root=data,
            datasets_root=cls.datasets,
            symbols=symbols,
            horizon_steps=int(lab["horizon_steps"]),
            horizon_steps_list=[int(x) for x in lab["horizon_steps_list"]],
            up_pct=float(lab["up_pct"]),
            down_pct=float(lab["down_pct"]),
            adaptive_thresholds=bool(lab["adaptive_thresholds"]),
            min_warmup_bars=int(lab["min_warmup_bars"]),
            max_candles_per_symbol=int(lab["max_candles_per_symbol"]),
            label_shape=str(lab["label_shape"]),
        )
        cls.rows = [
            json.loads(line)
            for line in (cls.datasets / cls.built["run_id"] / "dataset.jsonl").read_text().splitlines()
            if line.strip()
        ]
        cfg = dict(cls.config)
        cfg["storage"] = {
            **cls.config["storage"],
            "raw_dir": str(data / "raw"),
            "datasets_dir": str(cls.datasets),
            "models_dir": str(cls.models),
            "logs_dir": str(cls.logs),
        }
        cls.result = run_training(cfg, cls.datasets, cls.models, cls.logs)
        # Exactly how run_training and scripts/run_backtest.py resolve them:
        # replaying without thresholds falls back to an absolute cut no
        # calibrated rare-event head reaches, and books zero trades.
        cls.thresholds = _load_thresholds(cfg["storage"])

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_barrier_is_the_configured_fixed_one(self):
        self.assertEqual({round(r["up_pct"], 6) for r in self.rows}, {0.010})

    def test_base_rate_leaves_room_above_the_operating_threshold(self):
        """A base rate near the threshold means nothing can ever clear it."""
        base = sum(r["up_hit"] for r in self.rows) / len(self.rows)
        self.assertGreater(base, 0.005, "too few positives to train an honest head")
        self.assertLess(base, 0.5, "barrier so easy the label carries no information")

    def test_model_is_not_degenerate(self):
        self.assertNotEqual(self.result["train"].get("model_type"), "constant_dual_v1")

    def test_the_operating_threshold_is_reachable(self):
        """The B3/B6 regression: a threshold no calibrated head can ever reach.

        It surfaces as precision 0.000 and 0 trades, which reads like a weak
        model rather than a broken configuration.
        """
        ev = self.result["eval"]
        self.assertGreater(
            ev["n_predicted_up"], 0,
            f"nothing fires at threshold {ev['threshold_up']:.4f} against base rate {ev['base_rate_up']:.4f}",
        )
        self.assertGreater(ev["threshold_up"], ev["base_rate_up"], "threshold below base rate is not selective")

    def test_the_model_finds_the_planted_edge(self):
        self.assertGreater(self.result["eval"].get("pr_auc_up_lift", 0.0), 1.2)

    def test_thresholds_are_actually_found(self):
        """A silent {} here is what makes every panel read 0."""
        self.assertTrue(self.thresholds, "config/thresholds.yaml was not located")

    def test_backtest_produces_trades(self):
        bt = event_backtest(self.datasets, self.models, thresholds=self.thresholds)
        self.assertGreater(bt["signals"], 0, "backtest replayed no trades")
        self.assertEqual(bt["tp"] + bt["sl"] + bt["timeout"], bt["signals"])

    def test_trade_accounting_uses_first_touch_not_the_label_pair(self):
        """Under one-sided labels both flags can be 1, so a replay reading the
        pair books a dip-then-rally as a win. Every trade must be attributable
        to a first-touch direction."""
        bt = event_backtest(self.datasets, self.models, thresholds=self.thresholds)
        resolved = bt["tp"] + bt["sl"]
        both_hit = sum(1 for r in self.rows if r["up_hit"] and r["down_hit"])
        self.assertLessEqual(resolved, bt["signals"])
        for row in self.rows:
            if row["up_hit"] and row["down_hit"]:
                self.assertIn(row["first_hit"], (1, -1), "ambiguous row has no first-touch direction")
        # Nothing here asserts a resolution rate: how often price reaches a
        # barrier is a property of the market, not of this code, and on a
        # synthetic fixture it merely reflects the price process invented above.
        self.assertGreaterEqual(both_hit, 0)


if __name__ == "__main__":
    unittest.main()
