from __future__ import annotations

from dataclasses import dataclass, field
from time import time

from amber.alerts.console import print_alert
from amber.common.types import SignalV1


@dataclass(slots=True)
class AlertRateLimiter:
    cooldown_sec: int = 0
    _last_emit_ts: dict[tuple[str, str], float] = field(default_factory=dict)

    def allow(self, signal: SignalV1, channel: str) -> bool:
        if self.cooldown_sec <= 0:
            return True
        key = (signal.symbol, channel)
        now = time()
        last = self._last_emit_ts.get(key)
        if last is not None and (now - last) < self.cooldown_sec:
            return False
        self._last_emit_ts[key] = now
        return True


def route_alert(signal: SignalV1, channels: list[str] | None = None, limiter: AlertRateLimiter | None = None) -> None:
    channels = channels or ["console"]
    limiter = limiter or AlertRateLimiter(cooldown_sec=0)
    for ch in channels:
        if not limiter.allow(signal, ch):
            continue
        if ch == "console":
            print_alert(signal)
        # telegram/discord channels are intentionally no-op placeholders for now.
