"""The backtest and the threshold sweep must describe the gate the scanner runs.

Both kept replaying the pump/dump gate for three weeks after the scanner moved
to `move`; the sweep could even offer a one-click apply of prob_lift_min 1.2-3.0,
which would have opened the live move gate from 9.55.
"""

import json
import tempfile
import unittest
from pathlib import Path

from amber.backtest.backtester import replay_move
from amber.backtest.tuning import sweep_thresholds

COST = 0.0009


def _row(sym, ret_1, first_hit=0, spread=2.0):
    return {"symbol": sym, "ret_1": ret_1, "first_hit": first_hit, "up_pct": 0.01, "down_pct": 0.01,
            "horizon_steps": 2, "spread_bps": spread}


class TestReplayMove(unittest.TestCase):
    def test_momentum_follows_the_alert_bar_and_fade_takes_the_other_side(self):
        # alert on an up bar; the next bar's forward window hits +1% first.
        rows = [_row("A", +0.002), _row("A", 0.0, first_hit=1)]
        pnls, counts, extra = replay_move(rows, [0.9, 0.0], move_min=0.5, spread_max=30, cost=COST)
        self.assertEqual(counts["TP"], 1)
        self.assertAlmostEqual(pnls[0], 0.01 - COST)
        self.assertAlmostEqual(extra["fade_total"], -0.01 - COST)

    def test_a_down_bar_is_traded_short(self):
        rows = [_row("A", -0.002), _row("A", 0.0, first_hit=-1)]
        pnls, counts, _ = replay_move(rows, [0.9, 0.0], move_min=0.5, spread_max=30, cost=COST)
        self.assertEqual(counts["TP"], 1)

    def test_the_move_gate_and_spread_filter_decide(self):
        rows = [_row("A", 0.002), _row("A", 0.0, first_hit=1),
                _row("B", 0.002, spread=99), _row("B", 0.0, first_hit=1)]
        pnls, _, _ = replay_move(rows, [0.4, 0.0, 0.9, 0.0], move_min=0.5, spread_max=30, cost=COST)
        self.assertEqual(pnls, [], "traded below the move gate or above the spread cap")

    def test_a_flat_alert_bar_is_not_traded(self):
        rows = [_row("A", 0.0), _row("A", 0.0, first_hit=1)]
        pnls, _, extra = replay_move(rows, [0.9, 0.0], move_min=0.5, spread_max=30, cost=COST)
        self.assertEqual((pnls, extra["alerts_flat_bar"]), ([], 1))


class TestSweepOnMoveModels(unittest.TestCase):
    def test_the_pump_dump_sweep_refuses_to_tune_a_move_model(self):
        with tempfile.TemporaryDirectory() as td:
            models = Path(td) / "models"
            run = models / "model_1"
            run.mkdir(parents=True)
            (run / "model.json").write_text(json.dumps({
                "model_type": "x", "heads": {h: {"weights": {}, "bias": 0.0} for h in ("move", "pump", "dump")},
            }))
            (models / "registry.json").write_text(json.dumps({"model_run_id": "model_1"}))
            res = sweep_thresholds(models, Path(td) / "datasets")
            self.assertEqual(res["status"], "not_applicable")
            self.assertNotIn("best", res, "offered a threshold to apply")


if __name__ == "__main__":
    unittest.main()
