from __future__ import annotations

import logging
import os
import time

from amber.common.types import SignalV1
from amber.signals.explain import to_human_explanation

logger = logging.getLogger(__name__)

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

TOKEN_ENV = "AMBER_TG_TOKEN"
CHAT_ENV = "AMBER_TG_CHAT"
_RETRIES = 3


def send_telegram_alert(
    signal: SignalV1,
    bot_token: str | None = None,
    chat_id: str | None = None,
    *,
    timeout: float = 10.0,
    client: object | None = None,
) -> bool:
    """Deliver a signal alert via the Telegram Bot API. Returns True on success."""
    return send_telegram_text(
        f"AMBER {signal.symbol}\n{to_human_explanation(signal)}",
        bot_token=bot_token,
        chat_id=chat_id,
        timeout=timeout,
        client=client,
    )


def send_telegram_text(
    text: str,
    bot_token: str | None = None,
    chat_id: str | None = None,
    *,
    timeout: float = 10.0,
    client: object | None = None,
) -> bool:
    """Deliver a plain-text message (used for ops/health pushes, audit A6).

    Credentials come from arguments or the AMBER_TG_TOKEN / AMBER_TG_CHAT
    environment variables — never from code or config files.
    """
    token = bot_token or os.environ.get(TOKEN_ENV)
    chat = chat_id or os.environ.get(CHAT_ENV)
    if not token or not chat:
        logger.warning("telegram alert skipped: %s/%s not configured", TOKEN_ENV, CHAT_ENV)
        return False
    if client is None:
        if httpx is None:
            logger.error("telegram alert skipped: httpx is not installed")
            return False
        client = httpx.Client(timeout=timeout)

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for attempt in range(1, _RETRIES + 1):
        try:
            resp = client.post(url, json={"chat_id": chat, "text": text})  # type: ignore[attr-defined]
            if resp.status_code == 200:
                return True
            if resp.status_code == 429:
                retry_after = 1.0
                try:
                    retry_after = float(resp.json().get("parameters", {}).get("retry_after", 1.0))
                except Exception:
                    pass
                time.sleep(min(retry_after, 10.0))
                continue
            logger.warning("telegram send failed status=%s body=%s", resp.status_code, resp.text[:200])
        except Exception as exc:
            logger.warning("telegram send error attempt=%s err=%s", attempt, exc)
            time.sleep(min(2.0 ** attempt, 10.0))
    return False
