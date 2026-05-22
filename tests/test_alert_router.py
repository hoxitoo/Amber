import io
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone

from amber.alerts.router import AlertRateLimiter, route_alert
from amber.common.types import SignalExplanation, SignalV1


def _sig() -> SignalV1:
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


class TestAlertRouter(unittest.TestCase):
    def test_rate_limiter_blocks_duplicate_within_cooldown(self):
        signal = _sig()
        limiter = AlertRateLimiter(cooldown_sec=60)
        buff = io.StringIO()
        with redirect_stdout(buff):
            route_alert(signal, channels=["console"], limiter=limiter)
            route_alert(signal, channels=["console"], limiter=limiter)
        out = buff.getvalue().strip().splitlines()
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
