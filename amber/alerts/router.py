from __future__ import annotations

from amber.alerts.console import print_alert
from amber.common.types import SignalV1


def route_alert(signal: SignalV1, channels: list[str] | None = None) -> None:
    channels = channels or ["console"]
    for ch in channels:
        if ch == "console":
            print_alert(signal)
        # telegram/discord channels are intentionally no-op placeholders for now.
