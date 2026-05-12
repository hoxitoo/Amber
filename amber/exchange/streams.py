from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

try:
    import websockets
except Exception:  # pragma: no cover
    websockets = None


MessageHandler = Callable[[dict], Awaitable[None]]


class BybitWSClient:
    """Minimal resilient WS client for public Bybit linear stream.

    - auto reconnect
    - exponential backoff
    - topic subscription callback dispatch
    """

    def __init__(self, ws_url: str, topics: list[str], handler: MessageHandler) -> None:
        self.ws_url = ws_url
        self.topics = topics
        self.handler = handler
        self._stop = False

    async def run_forever(self) -> None:
        if websockets is None:
            raise RuntimeError("websockets dependency is not available")

        backoff = 1.0
        while not self._stop:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:  # type: ignore[attr-defined]
                    sub = {"op": "subscribe", "args": self.topics}
                    await ws.send(json.dumps(sub))
                    logger.info("WS connected, subscribed topics=%s", len(self.topics))
                    backoff = 1.0

                    async for raw in ws:
                        try:
                            payload = json.loads(raw)
                        except Exception:
                            continue
                        await self.handler(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("WS connection error: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    def stop(self) -> None:
        self._stop = True
