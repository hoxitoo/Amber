from __future__ import annotations

from amber.common.types import SignalV1


def send_discord_alert(signal: SignalV1, webhook_url: str | None = None) -> None:
    """Discord placeholder transport."""
    _ = (signal, webhook_url)
