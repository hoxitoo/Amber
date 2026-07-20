from __future__ import annotations

import logging
from time import time

from amber.alerts.console import print_alert
from amber.alerts.discord import send_discord_alert
from amber.alerts.telegram import send_telegram_alert
from amber.common.types import SignalV1
from amber.storage.state_store import StateStore

logger = logging.getLogger(__name__)

KNOWN_CHANNELS = ("console", "telegram", "discord")


class AlertRateLimiter:
    """Per (symbol, channel) cooldown. With a `store`, the state survives across
    scanner invocations — a batch process would otherwise reset it every run."""

    def __init__(
        self,
        cooldown_sec: int = 0,
        store: StateStore | None = None,
        state_key: str = "alert_limiter",
    ) -> None:
        self.cooldown_sec = cooldown_sec
        self.store = store
        self.state_key = state_key
        self._last_emit_ts: dict[str, float] = {}
        if store is not None:
            try:
                self._last_emit_ts = {k: float(v) for k, v in store.get(state_key).items()}
            except Exception:
                logger.warning("could not load alert limiter state key=%s; starting fresh", state_key)

    def allow(self, signal: SignalV1, channel: str) -> bool:
        if self.cooldown_sec <= 0:
            return True
        key = f"{signal.symbol}|{channel}"
        now = time()
        last = self._last_emit_ts.get(key)
        if last is not None and (now - last) < self.cooldown_sec:
            return False
        self._last_emit_ts[key] = now
        if self.store is not None:
            self.store.set(self.state_key, self._last_emit_ts)
        return True


def route_alert(signal: SignalV1, channels: list[str] | None = None, limiter: AlertRateLimiter | None = None) -> None:
    channels = channels or ["console"]
    limiter = limiter or AlertRateLimiter(cooldown_sec=0)
    for ch in channels:
        if ch not in KNOWN_CHANNELS:
            logger.warning("unknown alert channel %r (known: %s); alert not delivered", ch, ", ".join(KNOWN_CHANNELS))
            continue
        if not limiter.allow(signal, ch):
            continue
        if ch == "console":
            print_alert(signal)
        elif ch == "telegram":
            send_telegram_alert(signal)
        elif ch == "discord":
            send_discord_alert(signal)
