import unittest

from amber.exchange.normalizer import gap_fill
from amber.exchange.schemas import NormalizedRow


class TestGapFill(unittest.TestCase):
    def test_gap_fill_generates_synthetic_rows(self):
        last = NormalizedRow(
            ts=0,
            symbol="BTCUSDT",
            tf="1m",
            open=100,
            high=100,
            low=100,
            close=100,
            volume=1,
            bid=100,
            ask=100.1,
            oi=10,
            funding=0.0,
            is_synthetic=False,
        )
        rows = gap_fill(last, next_ts=180000, step_ms=60000)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r.is_synthetic for r in rows))
        self.assertTrue(all(r.volume == 0 for r in rows))


if __name__ == "__main__":
    unittest.main()
