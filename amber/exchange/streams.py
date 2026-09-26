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


def stream_kind(topic: str) -> str:
    """'kline.1.BTCUSDT' -> 'kline', 'allLiquidation.BTCUSDT' -> 'allLiquidation'."""
    return topic.split(".", 1)[0] if "." in topic else topic


def subscribe_requests(topics: list[str]) -> list[dict]:
    """One subscribe request per stream kind, each tagged with its kind as req_id.

    Everything used to go out as a single request. Bybit's docs do not say, for
    linear perpetuals, whether one invalid topic fails the whole request or only
    itself — and the client never read the response, so a rejection was silent.
    Adding a new stream to that single request therefore risked the candle
    stream the entire system depends on, with no log line to show for it.

    Split by kind, a rejected stream can only take itself down. `req_id` is
    echoed back by Bybit, so each response names the group it answers.
    Order is preserved so the core streams are requested first.
    """
    groups: dict[str, list[str]] = {}
    for topic in topics:
        groups.setdefault(stream_kind(topic), []).append(topic)
    return [{"op": "subscribe", "req_id": kind, "args": args} for kind, args in groups.items()]


def is_subscribe_response(payload: dict) -> bool:
    return payload.get("op") == "subscribe"


def log_subscribe_response(payload: dict) -> bool:
    """Log a subscribe acknowledgement; return whether it succeeded.

    A failure is a WARNING naming the stream, because the alternative — which
    was the behaviour until now — is a stream that quietly delivers nothing
    while everything else looks healthy.
    """
    group = payload.get("req_id") or "?"
    if payload.get("success"):
        logger.info("WS subscribe ok stream=%s", group)
        return True
    logger.warning(
        "WS subscribe REJECTED stream=%s ret_msg=%r — this stream will deliver no data; "
        "other streams were requested separately and are unaffected",
        group, payload.get("ret_msg", ""),
    )
    return False


class BybitWSClient:
    """Minimal resilient WS client for public Bybit linear stream.

    - auto reconnect
    - exponential backoff
    - one subscribe request per stream kind, each acknowledgement logged
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
                    requests = subscribe_requests(self.topics)
                    for req in requests:
                        await ws.send(json.dumps(req))
                    logger.info(
                        "WS connected, subscribed topics=%s in %s requests (%s)",
                        len(self.topics), len(requests), ", ".join(r["req_id"] for r in requests),
                    )
                    backoff = 1.0

                    async for raw in ws:
                        try:
                            payload = json.loads(raw)
                        except Exception:
                            continue
                        # Acknowledgements are control messages, not market data.
                        # They used to be forwarded and written to ws_raw/unknown.
                        if isinstance(payload, dict) and is_subscribe_response(payload):
                            log_subscribe_response(payload)
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
