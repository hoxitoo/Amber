"""Subscriptions are isolated per stream, and every acknowledgement is read.

The client used to send every topic in one subscribe request and never read the
reply. Bybit's documentation does not say, for linear perpetuals, whether one
invalid topic fails the whole request — so adding a stream risked the candle
feed the entire system depends on, and a rejection would have left no trace.
"""

import asyncio
import json
import unittest

from amber.exchange import streams
from amber.exchange.streams import (
    BybitWSClient,
    is_subscribe_response,
    log_subscribe_response,
    stream_kind,
    subscribe_requests,
)

TOPICS = [
    "kline.1.BTCUSDT", "kline.1.ETHUSDT",
    "tickers.BTCUSDT", "tickers.ETHUSDT",
    "publicTrade.BTCUSDT", "publicTrade.ETHUSDT",
    "allLiquidation.BTCUSDT", "allLiquidation.ETHUSDT",
]


class TestGrouping(unittest.TestCase):
    def test_stream_kind(self):
        self.assertEqual(stream_kind("kline.1.BTCUSDT"), "kline")
        self.assertEqual(stream_kind("allLiquidation.BTCUSDT"), "allLiquidation")
        self.assertEqual(stream_kind("weird"), "weird")

    def test_one_request_per_stream_kind(self):
        reqs = subscribe_requests(TOPICS)
        self.assertEqual([r["req_id"] for r in reqs], ["kline", "tickers", "publicTrade", "allLiquidation"])
        for r in reqs:
            self.assertEqual(r["op"], "subscribe")

    def test_liquidations_never_share_a_request_with_candles(self):
        """The protection itself: a rejected liquidation request must not be
        able to take klines down with it, whatever Bybit does with a batch."""
        for req in subscribe_requests(TOPICS):
            kinds = {stream_kind(t) for t in req["args"]}
            self.assertEqual(len(kinds), 1, f"request {req['req_id']} mixes {kinds}")

    def test_every_topic_is_requested_exactly_once(self):
        sent = [t for r in subscribe_requests(TOPICS) for t in r["args"]]
        self.assertEqual(sorted(sent), sorted(TOPICS))

    def test_core_streams_are_requested_first(self):
        self.assertEqual(subscribe_requests(TOPICS)[0]["req_id"], "kline")


class TestAcknowledgements(unittest.TestCase):
    def test_recognises_a_subscribe_reply(self):
        self.assertTrue(is_subscribe_response({"op": "subscribe", "success": True}))
        self.assertFalse(is_subscribe_response({"topic": "kline.1.BTCUSDT", "data": []}))

    def test_a_rejection_is_a_warning_naming_the_stream(self):
        with self.assertLogs(streams.logger, level="WARNING") as cm:
            ok = log_subscribe_response({"op": "subscribe", "success": False, "req_id": "allLiquidation",
                                         "ret_msg": "error:handler not found"})
        self.assertFalse(ok)
        self.assertIn("allLiquidation", cm.output[0])
        self.assertIn("handler not found", cm.output[0])

    def test_success_is_logged_at_info(self):
        with self.assertLogs(streams.logger, level="INFO") as cm:
            ok = log_subscribe_response({"op": "subscribe", "success": True, "req_id": "kline"})
        self.assertTrue(ok)
        self.assertIn("kline", cm.output[0])


class _FakeWS:
    """Just enough of a websocket: records sends, then yields scripted frames."""

    def __init__(self, frames: list[dict]) -> None:
        self.sent: list[dict] = []
        self._frames = [json.dumps(f) for f in frames]

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._frames:
            raise StopAsyncIteration
        return self._frames.pop(0)


class TestClientLoop(unittest.TestCase):
    def _run(self, frames: list[dict]) -> tuple[_FakeWS, list[dict]]:
        ws = _FakeWS(frames)
        received: list[dict] = []

        class _Conn:
            async def __aenter__(self_inner):
                return ws

            async def __aexit__(self_inner, *exc):
                client.stop()  # one connection, then leave the loop
                return False

        class _FakeWebsockets:
            @staticmethod
            def connect(*_a, **_k):
                return _Conn()

        async def handler(payload: dict) -> None:
            received.append(payload)

        client = BybitWSClient("wss://example", TOPICS, handler)
        original = streams.websockets
        streams.websockets = _FakeWebsockets
        try:
            asyncio.run(client.run_forever())
        finally:
            streams.websockets = original
        return ws, received

    def test_each_stream_is_requested_separately(self):
        ws, _ = self._run([])
        self.assertEqual([m["req_id"] for m in ws.sent], ["kline", "tickers", "publicTrade", "allLiquidation"])

    def test_acknowledgements_do_not_reach_the_data_handler(self):
        """They were forwarded and written to ws_raw/unknown as if they were data."""
        kline = {"topic": "kline.1.BTCUSDT", "data": []}
        _, received = self._run([
            {"op": "subscribe", "success": True, "req_id": "kline"},
            {"op": "subscribe", "success": False, "req_id": "allLiquidation", "ret_msg": "nope"},
            kline,
        ])
        self.assertEqual(received, [kline])

    def test_a_rejected_stream_leaves_data_from_the_others_flowing(self):
        kline = {"topic": "kline.1.BTCUSDT", "data": [{"close": "1"}]}
        with self.assertLogs(streams.logger, level="WARNING"):
            _, received = self._run([
                {"op": "subscribe", "success": False, "req_id": "allLiquidation", "ret_msg": "rejected"},
                kline,
            ])
        self.assertEqual(received, [kline])


if __name__ == "__main__":
    unittest.main()
