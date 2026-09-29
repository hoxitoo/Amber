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

    def test_a_move_signal_leads_with_the_move_probability_not_direction(self):
        s = SignalV1(
            signal_id="s2", event_ts=datetime.now(timezone.utc), symbol="ETHUSDT",
            horizon_min=15, target_up_pct=0.01, target_down_pct=0.01,
            prob_move_raw=0.8, prob_move_calibrated=0.73,
            prob_up_raw=0.6, prob_down_raw=0.2, prob_up_calibrated=0.61, prob_down_calibrated=0.18,
            regime="unknown", market_context={},
            explanation=SignalExplanation(top_feature_impacts=[{"range_atr_14": 1.2}],
                                          rule_trace=[{"rule": "directional_score", "value": 0.43}]),
            model_version="m", config_version="v1",
        )
        line = to_human_explanation(s)
        self.assertIn("P=0.73", line)
        self.assertIn("1.0%", line)
        self.assertIn("15 мин", line)
        self.assertIn("направление не прогнозируется", line)
        for directional in ("up_cal", "down_cal", "dir="):
            self.assertNotIn(directional, line)


if __name__ == "__main__":
    unittest.main()
