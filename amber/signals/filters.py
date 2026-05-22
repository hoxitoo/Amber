from __future__ import annotations

from dataclasses import dataclass, field
from time import time

from amber.common.types import SignalV1


def directional_score(signal: SignalV1) -> float:
    return signal.prob_up_calibrated - signal.prob_down_calibrated


def spread_bps(signal: SignalV1) -> float:
    bid = float(signal.market_context.get("bid", 0.0))
    ask = float(signal.market_context.get("ask", 0.0))
    mid = float(signal.market_context.get("mid_price", 0.0))
    if bid <= 0 or ask <= 0 or mid <= 0:
        ctx_spread = signal.market_context.get("spread_bps")
        return float(ctx_spread) if ctx_spread is not None else 0.0
    return ((ask - bid) / mid) * 10_000


@dataclass(slots=True)
class SignalGate:
    cooldown_sec: int
    concurrent_limit: int
    last_emit_ts: dict[str, float] = field(default_factory=dict)

    def allow(self, signal: SignalV1) -> bool:
        if len(self.last_emit_ts) >= self.concurrent_limit and signal.symbol not in self.last_emit_ts:
            return False
        now = time()
        last = self.last_emit_ts.get(signal.symbol)
        if last is not None and (now - last) < self.cooldown_sec:
            return False
        self.last_emit_ts[signal.symbol] = now
        return True


def passes_thresholds(
    signal: SignalV1,
    up_min: float,
    down_min: float,
    directional_min: float,
    spread_max_bps: float,
) -> bool:
    prob_ok = signal.prob_up_calibrated >= up_min or signal.prob_down_calibrated >= down_min
    dir_ok = abs(directional_score(signal)) >= directional_min
    spread_ok = spread_bps(signal) <= spread_max_bps
    return prob_ok and dir_ok and spread_ok
