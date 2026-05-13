import unittest

from amber.features.online import FeatureEngine


class TestFeatureEngine(unittest.TestCase):
    def test_engine_emits_spread_and_returns(self):
        eng = FeatureEngine()
        for i in range(25):
            out = eng.update(
                {
                    "symbol": "BTCUSDT",
                    "ts": i,
                    "close": 100 + i,
                    "volume": 10 + i,
                    "oi": 1000 + i,
                    "funding": 0.0001,
                    "bid": 100 + i - 0.05,
                    "ask": 100 + i + 0.05,
                }
            )

        self.assertIn("ret_20", out)
        self.assertIn("vol_z_20", out)
        self.assertIn("spread_bps", out)
        self.assertGreater(out["spread_bps"], 0.0)


if __name__ == "__main__":
    unittest.main()
