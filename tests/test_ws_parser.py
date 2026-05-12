import unittest

from amber.exchange.normalizer import BybitNormalizer


class TestWSParser(unittest.TestCase):
    def test_parse_kline_payload(self):
        payload = {
            "topic": "kline.1.BTCUSDT",
            "data": [
                {
                    "symbol": "BTCUSDT",
                    "interval": "1",
                    "start": 1700000000000,
                    "open": "100",
                    "high": "101",
                    "low": "99",
                    "close": "100.5",
                    "volume": "42",
                }
            ],
        }
        c = BybitNormalizer.candle_from_ws(payload)
        self.assertIsNotNone(c)
        self.assertEqual(c.symbol, "BTCUSDT")
        self.assertEqual(c.tf, "1m")
        self.assertEqual(c.ts, 1700000000000)


if __name__ == "__main__":
    unittest.main()
