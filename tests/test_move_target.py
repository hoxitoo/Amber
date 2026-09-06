"""The primary target is movement, not direction (roadmap D10).

Direction was measured on live data and found absent: precision 0.590 against a
0.587 base rate, +0.3pp over naively saying "up". The model now gates on
`move` — will price travel the barrier at all — while the pump/dump heads are
kept for the deferred revisit.

Two things have to hold at once, and they pull in opposite directions: the new
target must actually drive the system, and a model without it (an older
artifact, a rollback) must keep working rather than crash or fire on
everything.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from amber.common.types import SignalExplanation, SignalV1
from amber.labeling.events import move_label
from amber.models.features import MODEL_FEATURES
from amber.models.train import _fit_dual
from amber.signals.filters import passes_thresholds

BASE_SIGNAL = {
    "signal_id": "sig_test",
    "event_ts": 1_700_000_000_000,
    "symbol": "AAAUSDT",
    "horizon_min": 15,
    "target_up_pct": 0.01,
    "target_down_pct": 0.01,
    "regime": "unknown",
    "model_version": "lightgbm_dual_v1",
    "config_version": "v1",
}


def _signal(**kw) -> SignalV1:
    payload = {
        **BASE_SIGNAL,
        "prob_up_raw": 0.2,
        "prob_down_raw": 0.2,
        "prob_up_calibrated": 0.2,
        "prob_down_calibrated": 0.2,
        "market_context": {"spread_bps": 2.0},
        "explanation": SignalExplanation(top_feature_impacts=[], rule_trace=[]),
        **kw,
    }
    return SignalV1(**payload)


class TestMoveLabel(unittest.TestCase):
    def test_move_label_is_used_when_present(self):
        self.assertEqual(move_label({"move_hit": 1, "up_hit": 0, "down_hit": 0}), 1)
        self.assertEqual(move_label({"move_hit": 0, "up_hit": 1, "down_hit": 0}), 0)

    def test_move_label_is_derived_for_older_datasets(self):
        """Defaulting a missing move_hit to 0 would hand the head a single class
        and collapse it to a constant, which is how a previous target change
        silently produced constant_dual_v1."""
        self.assertEqual(move_label({"up_hit": 1, "down_hit": 0}), 1)
        self.assertEqual(move_label({"up_hit": 0, "down_hit": 1}), 1)
        self.assertEqual(move_label({"up_hit": 0, "down_hit": 0}), 0)


class TestModelHasAMoveHead(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = random.Random(3)
        rows = []
        for i in range(600):
            row = {name: rng.gauss(0, 1) for name in MODEL_FEATURES}
            hot = row["vol_z_20"] > 1.0
            up = int(hot and rng.random() < 0.5)
            down = int(hot and not up)
            row.update({
                "up_hit": up,
                "down_hit": down,
                "move_hit": int(bool(up or down)),
                "first_hit": 1 if up else (-1 if down else 0),
                "horizon_steps": 15,
            })
            rows.append(row)
        cls.model = _fit_dual(rows)

    def test_all_three_heads_are_fitted(self):
        self.assertEqual(set(self.model["heads"]), {"move", "pump", "dump"})
        self.assertEqual(self.model["primary_target"], "move")

    def test_direction_heads_are_kept_for_the_deferred_revisit(self):
        """D10 is a deferral, not a closed question — deleting these would mean
        rebuilding the machinery to reopen it."""
        for head in ("pump", "dump"):
            self.assertIn("label_rate", self.model["heads"][head])

    def test_move_base_rate_is_at_least_either_direction(self):
        rates = {k: self.model["heads"][k]["label_rate"] for k in ("move", "pump", "dump")}
        self.assertGreaterEqual(rates["move"], rates["pump"])
        self.assertGreaterEqual(rates["move"], rates["dump"])


class TestGate(unittest.TestCase):
    def test_move_gate_ignores_the_directional_filter(self):
        """A two-sided volatile setup is exactly what a volatility scanner should
        surface, and the directional filter would suppress it."""
        sig = _signal(prob_move_calibrated=0.9, prob_up_calibrated=0.45, prob_down_calibrated=0.45)

        self.assertTrue(
            passes_thresholds(sig, up_min=0.9, down_min=0.9, directional_min=0.5,
                              spread_max_bps=30.0, move_min=0.5)
        )

    def test_move_gate_rejects_below_threshold(self):
        sig = _signal(prob_move_calibrated=0.2)
        self.assertFalse(
            passes_thresholds(sig, up_min=0.0, down_min=0.0, directional_min=0.0,
                              spread_max_bps=30.0, move_min=0.5)
        )

    def test_move_gate_still_enforces_spread(self):
        sig = _signal(prob_move_calibrated=0.99, market_context={"spread_bps": 90.0})
        self.assertFalse(
            passes_thresholds(sig, up_min=0.0, down_min=0.0, directional_min=0.0,
                              spread_max_bps=30.0, move_min=0.5)
        )

    def test_signal_without_a_move_probability_falls_back(self):
        """Signals from a model predating the change must still be gateable."""
        sig = _signal(prob_up_calibrated=0.3, prob_down_calibrated=0.3)
        self.assertIsNone(sig.prob_move_calibrated)
        self.assertTrue(
            passes_thresholds(sig, up_min=0.9, down_min=0.9, directional_min=0.9,
                              spread_max_bps=30.0, move_min=0.5)
        )

    def test_legacy_gate_is_untouched_without_move_min(self):
        """No move_min means an older model, and its behaviour must not change."""
        sig = _signal(prob_up_calibrated=0.45, prob_down_calibrated=0.45)
        self.assertFalse(
            passes_thresholds(sig, up_min=0.4, down_min=0.4, directional_min=0.5, spread_max_bps=30.0)
        )
        self.assertTrue(
            passes_thresholds(sig, up_min=0.4, down_min=0.4, directional_min=0.0, spread_max_bps=30.0)
        )


class TestOutcomeConfirmation(unittest.TestCase):
    """A movement signal graded on a pump-only outcome counts every correctly
    called downward move as a miss."""

    def _index(self, lows_dip: bool):
        from amber.monitoring.quality_report import _CandleIndex

        td = Path(tempfile.mkdtemp())
        d = td / "normalized" / "AAAUSDT"
        d.mkdir(parents=True)
        rows = []
        for i in range(40):
            price = 100.0 if i < 5 else (98.0 if lows_dip else 100.0)
            rows.append({
                "ts": 1_700_000_000_000 + i * 60_000,
                "close": price, "high": price * 1.001, "low": price * 0.999,
            })
        (d / "part-000.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return _CandleIndex(td)

    def test_a_downward_move_counts_only_when_both_directions_are_scored(self):
        from amber.monitoring.quality_report import _confirmed_outcome

        index = self._index(lows_dip=True)
        args = (index, "AAAUSDT", 1_700_000_000_000, 15, 0.01)

        self.assertEqual(_confirmed_outcome(*args), 0)  # pump-only: a miss
        self.assertEqual(_confirmed_outcome(*args, both_directions=True), 1)

    def test_a_flat_market_is_a_miss_either_way(self):
        from amber.monitoring.quality_report import _confirmed_outcome

        index = self._index(lows_dip=False)
        args = (index, "AAAUSDT", 1_700_000_000_000, 15, 0.01)

        self.assertEqual(_confirmed_outcome(*args), 0)
        self.assertEqual(_confirmed_outcome(*args, both_directions=True), 0)


if __name__ == "__main__":
    unittest.main()
