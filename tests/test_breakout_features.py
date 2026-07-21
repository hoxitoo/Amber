"""Tests for the breakout / momentum-precursor feature pack."""

import unittest

from amber.features.online import FeatureEngine
from amber.models.features import MODEL_FEATURES


def _row(symbol, ts, close, high=None, low=None, volume=10.0, oi=1000.0):
    return {
        "symbol": symbol,
        "ts": ts,
        "close": close,
        "high": high if high is not None else close,
        "low": low if low is not None else close,
        "volume": volume,
        "oi": oi,
        "funding": 0.0001,
        "bid": close - 0.01,
        "ask": close + 0.01,
    }


class TestBreakoutFeatures(unittest.TestCase):
    def test_all_model_features_are_emitted(self):
        eng = FeatureEngine()
        out = {}
        for i in range(80):
            out = eng.update(_row("BTCUSDT", i, 100 + i * 0.01))
        for name in MODEL_FEATURES:
            self.assertIn(name, out, f"missing feature {name}")

    def test_breakout_up_flag_fires_on_new_high(self):
        eng = FeatureEngine()
        out = {}
        # 30 bars ranging tightly around 100, then a bar that breaks above the range
        for i in range(30):
            out = eng.update(_row("BTCUSDT", i, 100.0, high=100.2, low=99.8))
        self.assertEqual(out["breakout_up_20"], 0.0)
        out = eng.update(_row("BTCUSDT", 30, 101.0, high=101.5, low=100.0))
        self.assertEqual(out["breakout_up_20"], 1.0)
        # close sits near the top of the 20-bar range (within ~1% of the window high)
        self.assertGreater(out["dist_to_high_20"], -0.01)

    def test_breakout_down_flag_fires_on_new_low(self):
        eng = FeatureEngine()
        out = {}
        for i in range(30):
            out = eng.update(_row("ETHUSDT", i, 100.0, high=100.2, low=99.8))
        out = eng.update(_row("ETHUSDT", 30, 99.0, high=100.0, low=98.5))
        self.assertEqual(out["breakout_dn_20"], 1.0)

    def test_volume_surge_detected(self):
        eng = FeatureEngine()
        out = {}
        for i in range(30):
            out = eng.update(_row("BTCUSDT", i, 100.0, volume=10.0))
        self.assertAlmostEqual(out["vol_ratio_20"], 1.0, places=1)
        out = eng.update(_row("BTCUSDT", 30, 100.0, volume=100.0))  # 10x volume spike
        self.assertGreater(out["vol_ratio_20"], 5.0)
        self.assertGreater(out["vol_accel"], 1.5)

    def test_squeeze_ratio_low_when_compressed(self):
        eng = FeatureEngine()
        out = {}
        # long stretch of tiny moves -> low short/long vol ratio is well defined
        price = 100.0
        for i in range(40):
            price *= 1 + (0.0001 if i % 2 else -0.0001)
            out = eng.update(_row("BTCUSDT", i, price))
        self.assertGreaterEqual(out["squeeze_ratio"], 0.0)
        self.assertLess(out["bb_width_20"], 0.01)

    def test_oi_roc_positive_when_oi_ramps(self):
        eng = FeatureEngine()
        out = {}
        for i in range(10):
            out = eng.update(_row("BTCUSDT", i, 100.0, oi=1000.0 + i * 100.0))
        self.assertGreater(out["oi_roc_5"], 0.0)

    def test_features_stable_on_short_history(self):
        eng = FeatureEngine()
        out = eng.update(_row("BTCUSDT", 0, 100.0))
        # no division by zero / no crash on the very first bar
        for name in MODEL_FEATURES:
            self.assertIsInstance(out[name], float)


if __name__ == "__main__":
    unittest.main()
