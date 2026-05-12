from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class MarketTick:
    ts: str
    symbol: str
    bid: float
    ask: float
    last: float
    volume_1m: float


class BybitClient:
    """Local stub client for deterministic development without live API calls."""

    def fetch_stub_tick(self, symbol: str) -> MarketTick:
        now = datetime.now(timezone.utc).isoformat()
        return MarketTick(
            ts=now,
            symbol=symbol,
            bid=100.0,
            ask=100.1,
            last=100.05,
            volume_1m=42.0,
        )

    def as_dict(self, tick: MarketTick) -> dict[str, Any]:
        return {
            "ts": tick.ts,
            "symbol": tick.symbol,
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "volume_1m": tick.volume_1m,
        }
