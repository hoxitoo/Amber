import unittest
from datetime import datetime, timezone

from amber.alerts.router import AlertRateLimiter
from amber.common.types import SignalExplanation, SignalV1


def _sig() -> SignalV1:
    return SignalV1(
        signal_id="x",
        event_ts=datetime.now(timezone.utc),
        symbol="BTCUSDT",
        horizon_min=5,
        target_up_pct=0.002,
        target_down_pct=0.002,
        prob_up_raw=0.8,
        prob_down_raw=0.2,
        prob_up_calibrated=0.75,
        prob_down_calibrated=0.2,
        regime="unknown",
        market_context={},
        explanation=SignalExplanation(),
        model_version="m",
        config_version="v1",
    )


class TestScannerAlerts(unittest.TestCase):
    def test_non_positive_cooldown_always_allows(self):
        lim = AlertRateLimiter(cooldown_sec=0)
        self.assertTrue(lim.allow(_sig(), "console"))
        self.assertTrue(lim.allow(_sig(), "console"))
        lim_neg = AlertRateLimiter(cooldown_sec=-5)
        self.assertTrue(lim_neg.allow(_sig(), "console"))

    def test_positive_cooldown_blocks_repeat(self):
        lim = AlertRateLimiter(cooldown_sec=60)
        self.assertTrue(lim.allow(_sig(), "console"))
        self.assertFalse(lim.allow(_sig(), "console"))


if __name__ == "__main__":
    unittest.main()
