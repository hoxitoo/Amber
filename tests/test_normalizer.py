import unittest

from amber.exchange.normalizer import BybitNormalizer
from amber.exchange.schemas import Candle, FundingRate, OpenInterest, Ticker


class TestNormalizer(unittest.TestCase):
    def test_to_normalized_uses_cached_market_state(self):
        n = BybitNormalizer()
        n.update_ticker(Ticker(ts=1, symbol="BTCUSDT", bid=100.0, ask=100.2, last=100.1))
        n.update_oi(OpenInterest(ts=1, symbol="BTCUSDT", oi=123.0))
        n.update_funding(FundingRate(ts=1, symbol="BTCUSDT", funding=0.001))

        row = n.to_normalized(Candle(ts=1, symbol="BTCUSDT", tf="1m", open=100, high=101, low=99, close=100.5, volume=42))
        self.assertEqual(row.bid, 100.0)
        self.assertEqual(row.ask, 100.2)
        self.assertEqual(row.oi, 123.0)
        self.assertEqual(row.funding, 0.001)


if __name__ == "__main__":
    unittest.main()
