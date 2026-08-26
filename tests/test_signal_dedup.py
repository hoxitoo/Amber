"""One candle must produce at most one signal."""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from amber.common.types import SignalExplanation, SignalV1
from amber.signals.filters import SignalGate
from amber.storage.state_store import StateStore


def sig(symbol="BTCUSDT", event_ts=None):
    return SignalV1(
        signal_id="x", event_ts=event_ts or datetime(2026, 8, 26, 14, 59, tzinfo=timezone.utc),
        symbol=symbol, horizon_min=15, target_up_pct=0.007, target_down_pct=0.007,
        prob_up_raw=0.8, prob_down_raw=0.2, prob_up_calibrated=0.35, prob_down_calibrated=0.19,
        regime="unknown", market_context={"bid": 100.0, "ask": 100.1, "mid_price": 100.05},
        explanation=SignalExplanation(), model_version="m", config_version="v1")


class TestOneSignalPerCandle(unittest.TestCase):
    def test_same_candle_never_emits_twice_even_after_cooldown(self):
        g = SignalGate(cooldown_sec=0, concurrent_limit=5)   # cooldown fully lapsed
        self.assertTrue(g.allow(sig()))
        self.assertFalse(g.allow(sig()), "same candle emitted a second time")
        self.assertFalse(g.allow(sig()))

    def test_a_new_candle_is_allowed(self):
        g = SignalGate(cooldown_sec=0, concurrent_limit=5)
        t = datetime(2026, 8, 26, 14, 59, tzinfo=timezone.utc)
        self.assertTrue(g.allow(sig(event_ts=t)))
        self.assertTrue(g.allow(sig(event_ts=t + timedelta(minutes=1))))

    def test_older_candle_is_rejected(self):
        g = SignalGate(cooldown_sec=0, concurrent_limit=5)
        t = datetime(2026, 8, 26, 14, 59, tzinfo=timezone.utc)
        self.assertTrue(g.allow(sig(event_ts=t)))
        self.assertFalse(g.allow(sig(event_ts=t - timedelta(minutes=5))))

    def test_symbols_are_tracked_independently(self):
        g = SignalGate(cooldown_sec=0, concurrent_limit=5)
        self.assertTrue(g.allow(sig("BTCUSDT")))
        self.assertTrue(g.allow(sig("ETHUSDT")))

    def test_dedup_survives_a_scanner_restart(self):
        with tempfile.TemporaryDirectory() as td:
            store = StateStore(Path(td))
            g1 = SignalGate(cooldown_sec=0, concurrent_limit=5, store=store)
            self.assertTrue(g1.allow(sig()))
            g2 = SignalGate(cooldown_sec=0, concurrent_limit=5, store=store)
            self.assertFalse(g2.allow(sig()), "duplicate slipped through after restart")

    def test_legacy_flat_state_still_loads(self):
        with tempfile.TemporaryDirectory() as td:
            store = StateStore(Path(td))
            store.set("signal_gate", {"BTCUSDT": 1.0})     # pre-change format
            g = SignalGate(cooldown_sec=0, concurrent_limit=5, store=store)
            self.assertEqual(g.last_emit_ts["BTCUSDT"], 1.0)
            self.assertTrue(g.allow(sig()))


if __name__ == "__main__":
    unittest.main()
