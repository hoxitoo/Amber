import logging
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from amber.alerts.router import AlertRateLimiter, route_alert
from amber.alerts.telegram import send_telegram_alert
from amber.common.types import SignalExplanation, SignalV1
from amber.signals.filters import SignalGate
from amber.storage.state_store import StateStore


def _sig(symbol: str = "BTCUSDT") -> SignalV1:
    return SignalV1(
        signal_id="x",
        event_ts=datetime.now(timezone.utc),
        symbol=symbol,
        horizon_min=5,
        target_up_pct=0.002,
        target_down_pct=0.002,
        prob_up_raw=0.8,
        prob_down_raw=0.2,
        prob_up_calibrated=0.75,
        prob_down_calibrated=0.2,
        regime="unknown",
        market_context={"bid": 100.0, "ask": 100.1, "mid_price": 100.05},
        explanation=SignalExplanation(),
        model_version="m",
        config_version="v1",
    )


class _FakeResponse:
    def __init__(self, status_code: int = 200, text: str = "ok") -> None:
        self.status_code = status_code
        self.text = text

    def json(self):
        return {}


class _FakeClient:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code
        self.calls: list[tuple[str, dict]] = []

    def post(self, url: str, json: dict) -> _FakeResponse:
        self.calls.append((url, json))
        return _FakeResponse(self.status_code)


class TestTelegramTransport(unittest.TestCase):
    def test_sends_message_with_credentials(self):
        client = _FakeClient()
        ok = send_telegram_alert(_sig(), bot_token="t0k", chat_id="123", client=client)
        self.assertTrue(ok)
        self.assertEqual(len(client.calls), 1)
        url, payload = client.calls[0]
        self.assertIn("bott0k/sendMessage", url)
        self.assertEqual(payload["chat_id"], "123")
        self.assertIn("BTCUSDT", payload["text"])

    def test_missing_credentials_returns_false_without_call(self):
        client = _FakeClient()
        import os

        old_token = os.environ.pop("AMBER_TG_TOKEN", None)
        old_chat = os.environ.pop("AMBER_TG_CHAT", None)
        try:
            ok = send_telegram_alert(_sig(), client=client)
        finally:
            if old_token:
                os.environ["AMBER_TG_TOKEN"] = old_token
            if old_chat:
                os.environ["AMBER_TG_CHAT"] = old_chat
        self.assertFalse(ok)
        self.assertEqual(client.calls, [])


class TestRouterChannels(unittest.TestCase):
    def test_unknown_channel_warns_instead_of_silence(self):
        with self.assertLogs("amber.alerts.router", level=logging.WARNING) as captured:
            route_alert(_sig(), channels=["carrier_pigeon"])
        self.assertTrue(any("unknown alert channel" in m for m in captured.output))


class TestPersistentState(unittest.TestCase):
    def test_rate_limiter_survives_process_restart(self):
        with tempfile.TemporaryDirectory() as td:
            store = StateStore(Path(td))
            first = AlertRateLimiter(cooldown_sec=3600, store=store)
            self.assertTrue(first.allow(_sig(), "console"))
            # "new process": fresh limiter instance backed by the same store
            second = AlertRateLimiter(cooldown_sec=3600, store=StateStore(Path(td)))
            self.assertFalse(second.allow(_sig(), "console"))

    def test_signal_gate_survives_restart_and_releases_slots(self):
        with tempfile.TemporaryDirectory() as td:
            store = StateStore(Path(td))
            gate = SignalGate(cooldown_sec=3600, concurrent_limit=2, slot_ttl_sec=3600, store=store)
            self.assertTrue(gate.allow(_sig("AAAUSDT")))
            self.assertTrue(gate.allow(_sig("BBBUSDT")))
            # limit reached -> new symbol blocked
            self.assertFalse(gate.allow(_sig("CCCUSDT")))
            # restart: same store, cooldown still applies to seen symbol
            gate2 = SignalGate(cooldown_sec=3600, concurrent_limit=2, slot_ttl_sec=3600, store=StateStore(Path(td)))
            self.assertFalse(gate2.allow(_sig("AAAUSDT")))
            # expired slots release capacity for new symbols
            gate3 = SignalGate(cooldown_sec=1, concurrent_limit=2, slot_ttl_sec=1, store=StateStore(Path(td)))
            gate3.last_emit_ts = {k: v - 10 for k, v in gate3.last_emit_ts.items()}
            self.assertTrue(gate3.allow(_sig("CCCUSDT")))


if __name__ == "__main__":
    unittest.main()
