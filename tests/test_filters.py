import unittest
from datetime import datetime, timezone

from amber.common.types import SignalExplanation, SignalV1
from amber.signals.filters import directional_score, passes_thresholds, spread_bps


class TestFilters(unittest.TestCase):
    def _sig(self) -> SignalV1:
        return SignalV1(
            signal_id="x",
            event_ts=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            horizon_min=5,
            target_up_pct=0.2,
            target_down_pct=0.2,
            prob_up_raw=0.8,
            prob_down_raw=0.2,
            prob_up_calibrated=0.75,
            prob_down_calibrated=0.2,
            regime="unknown",
            market_context={"bid": 100.0, "ask": 100.1, "mid_price": 100.05},
            explanation=SignalExplanation(),
            model_version="m",
            config_version="v1",
        )

    def test_directional_and_spread(self):
        s = self._sig()
        self.assertGreater(directional_score(s), 0.0)
        self.assertGreater(spread_bps(s), 0.0)
        self.assertTrue(
            passes_thresholds(
                s,
                up_min=0.65,
                down_min=0.65,
                directional_min=0.2,
                spread_max_bps=30.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
