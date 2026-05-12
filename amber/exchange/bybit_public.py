from __future__ import annotations

from datetime import datetime, timezone

from amber.exchange.schemas import Candle, FundingRate, OpenInterest, Ticker


class BybitPublicClient:
    """Stub public market client producing deterministic market snapshots.

    Contract matches future WS/REST implementation points.
    """

    def _now_ms(self) -> int:
        return int(datetime.now(timezone.utc).timestamp() * 1000)

    def get_candle(self, symbol: str, tf: str = "1m") -> Candle:
        ts = self._now_ms()
        return Candle(ts=ts, symbol=symbol, tf=tf, open=100.0, high=100.3, low=99.8, close=100.1, volume=42.0)

    def get_ticker(self, symbol: str) -> Ticker:
        ts = self._now_ms()
        return Ticker(ts=ts, symbol=symbol, bid=100.0, ask=100.2, last=100.1)

    def get_open_interest(self, symbol: str) -> OpenInterest:
        ts = self._now_ms()
        return OpenInterest(ts=ts, symbol=symbol, oi=1_000_000.0)

    def get_funding(self, symbol: str) -> FundingRate:
        ts = self._now_ms()
        return FundingRate(ts=ts, symbol=symbol, funding=0.0001)
