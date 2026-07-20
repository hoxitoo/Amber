import unittest

from amber.exchange.normalizer import BybitNormalizer, step_ms_for_tf


def _kline_payload(confirm: bool, start: int = 1700000000000, close: str = "100.5") -> dict:
    # Real Bybit v5 kline shape: symbol only in topic, data items carry confirm flag.
    return {
        "topic": "kline.1.BTCUSDT",
        "type": "snapshot",
        "ts": start + 30_000,
        "data": [
            {
                "start": start,
                "end": start + 60_000,
                "interval": "1",
                "open": "100",
                "high": "101",
                "low": "99",
                "close": close,
                "volume": "42",
                "turnover": "4200",
                "confirm": confirm,
                "timestamp": start + 30_000,
            }
        ],
    }


class TestWSParser(unittest.TestCase):
    def test_parse_confirmed_kline_payload(self):
        c = BybitNormalizer.candle_from_ws(_kline_payload(confirm=True))
        self.assertIsNotNone(c)
        self.assertEqual(c.symbol, "BTCUSDT")
        self.assertEqual(c.tf, "1m")
        self.assertEqual(c.ts, 1700000000000)
        self.assertEqual(c.close, 100.5)

    def test_unconfirmed_kline_is_skipped(self):
        self.assertIsNone(BybitNormalizer.candle_from_ws(_kline_payload(confirm=False)))

    def test_symbol_missing_from_data_is_fine(self):
        payload = _kline_payload(confirm=True)
        self.assertNotIn("symbol", payload["data"][0])
        c = BybitNormalizer.candle_from_ws(payload)
        self.assertEqual(c.symbol, "BTCUSDT")

    def test_non_kline_topic_returns_none(self):
        self.assertIsNone(BybitNormalizer.candle_from_ws({"topic": "tickers.BTCUSDT", "data": {}}))

    def test_ticker_snapshot_and_delta_merge(self):
        n = BybitNormalizer()
        handled = n.update_from_ws_ticker(
            {
                "topic": "tickers.BTCUSDT",
                "type": "snapshot",
                "ts": 1700000000000,
                "data": {
                    "symbol": "BTCUSDT",
                    "lastPrice": "100.1",
                    "bid1Price": "100.0",
                    "ask1Price": "100.2",
                    "fundingRate": "0.0001",
                    "openInterest": "123456",
                },
            }
        )
        self.assertTrue(handled)
        # delta with only funding change keeps previous bid/ask
        n.update_from_ws_ticker(
            {
                "topic": "tickers.BTCUSDT",
                "type": "delta",
                "ts": 1700000001000,
                "data": {"symbol": "BTCUSDT", "fundingRate": "0.0002"},
            }
        )
        t = n.cache.tickers["BTCUSDT"]
        self.assertEqual(t.bid, 100.0)
        self.assertEqual(t.ask, 100.2)
        self.assertEqual(n.cache.funding["BTCUSDT"].funding, 0.0002)
        self.assertEqual(n.cache.oi["BTCUSDT"].oi, 123456.0)

    def test_step_ms_for_tf(self):
        self.assertEqual(step_ms_for_tf("1m"), 60_000)
        self.assertEqual(step_ms_for_tf("5m"), 300_000)
        self.assertEqual(step_ms_for_tf("1h"), 3_600_000)
        self.assertEqual(step_ms_for_tf("weird"), 60_000)


if __name__ == "__main__":
    unittest.main()
