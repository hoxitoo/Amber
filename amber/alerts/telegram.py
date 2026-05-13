from __future__ import annotations

from amber.common.types import SignalV1


def send_telegram_alert(signal: SignalV1, bot_token: str | None = None, chat_id: str | None = None) -> None:
    """Telegram placeholder transport.

    Keep explicit interface so router integration remains stable when real delivery is added.
    """
    _ = (signal, bot_token, chat_id)
