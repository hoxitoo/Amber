from __future__ import annotations

from amber.common.types import SignalV1


def passes_thresholds(signal: SignalV1, up_min: float, down_min: float) -> bool:
    return signal.prob_up_calibrated >= up_min or signal.prob_down_calibrated >= down_min
