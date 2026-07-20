from __future__ import annotations

import logging
import os

from amber.common.types import SignalV1
from amber.signals.explain import to_human_explanation

logger = logging.getLogger(__name__)

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

WEBHOOK_ENV = "AMBER_DISCORD_WEBHOOK"


def send_discord_alert(
    signal: SignalV1,
    webhook_url: str | None = None,
    *,
    timeout: float = 10.0,
    client: object | None = None,
) -> bool:
    """Deliver an alert via a Discord webhook. Returns True on success."""
    url = webhook_url or os.environ.get(WEBHOOK_ENV)
    if not url:
        logger.warning("discord alert skipped: %s not configured", WEBHOOK_ENV)
        return False
    if client is None:
        if httpx is None:
            logger.error("discord alert skipped: httpx is not installed")
            return False
        client = httpx.Client(timeout=timeout)
    try:
        resp = client.post(url, json={"content": f"AMBER {signal.symbol}\n{to_human_explanation(signal)}"})  # type: ignore[attr-defined]
        if resp.status_code in (200, 204):
            return True
        logger.warning("discord send failed status=%s body=%s", resp.status_code, resp.text[:200])
    except Exception as exc:
        logger.warning("discord send error err=%s", exc)
    return False
