from __future__ import annotations

from amber.common.types import SignalV1


def to_human_explanation(signal: SignalV1) -> str:
    impacts = ", ".join([f"{list(x.keys())[0]}={list(x.values())[0]:.4f}" for x in signal.explanation.top_feature_impacts])
    return (
        f"{signal.symbol} | up_cal={signal.prob_up_calibrated:.3f} "
        f"down_cal={signal.prob_down_calibrated:.3f} | impacts: {impacts}"
    )
