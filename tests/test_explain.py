import unittest
from datetime import datetime, timezone

from amber.common.types import SignalExplanation, SignalV1
from amber.signals.explain import to_human_explanation


class TestExplain(unittest.TestCase):
    def test_to_human_explanation_renders_directional_and_top_n(self):
        s = SignalV1(
            signal_id="s1",
            event_ts=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            horizon_min=5,
            target_up_pct=0.2,
            target_down_pct=0.2,
            prob_up_raw=0.7,
            prob_down_raw=0.3,
            prob_up_calibrated=0.68,
            prob_down_calibrated=0.21,
            regime="unknown",
            market_context={},
            explanation=SignalExplanation(
                top_feature_impacts=[
                    {"a": 0.1},
                    {"b": -0.4},
                    {"c": 0.2},
                    {"d": 0.05},
                ],
                rule_trace=[{"rule": "directional_score", "value": 0.47}],
            ),
            model_version="m",
            config_version="v1",
        )
        line = to_human_explanation(s, top_n=2)
        self.assertIn("dir=+0.470", line)
        self.assertIn("b=-0.4000", line)
        self.assertIn("c=0.2000", line)
        self.assertNotIn("a=0.1000", line)


if __name__ == "__main__":
    unittest.main()
