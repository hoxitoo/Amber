from __future__ import annotations

import logging
from typing import Any

from amber.exchange.schemas import Candle, FundingRate, OpenInterest, Ticker

logger = logging.getLogger(__name__)

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

MAINNET_REST_URL = "https://api.bybit.com"
TESTNET_REST_URL = "https://api-testnet.bybit.com"


class BybitPublicClient:
    """REST client for Bybit v5 public market endpoints (linear category).

    Used for historical backfill and market-state snapshots; the live path uses
    the WS client in `amber/exchange/streams.py`.
    """

    def __init__(
        self,
        rest_url: str = MAINNET_REST_URL,
        *,
        category: str = "linear",
        timeout: float = 10.0,
        client: Any | None = None,
    ) -> None:
        if client is None:
            if httpx is None:
                raise RuntimeError("httpx dependency is not available; install requirements.txt")
            client = httpx.Client(base_url=rest_url, timeout=timeout)
        self._client = client
        self.category = category

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BybitPublicClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        resp = self._client.get(path, params=params)
        resp.raise_for_status()
        body = resp.json()
        if int(body.get("retCode", -1)) != 0:
            raise RuntimeError(f"Bybit API error path={path} retCode={body.get('retCode')} retMsg={body.get('retMsg')}")
        result = body.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"Bybit API malformed result path={path}")
        return result

    def get_klines(
        self,
        symbol: str,
        *,
        interval: str = "1",
        limit: int = 200,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[Candle]:
        """Fetch klines, returned oldest-first (Bybit sends newest-first)."""
        params: dict[str, Any] = {
            "category": self.category,
            "symbol": symbol,
            "interval": interval,
            "limit": max(1, min(int(limit), 1000)),
        }
        if start_ms is not None:
            params["start"] = int(start_ms)
        if end_ms is not None:
            params["end"] = int(end_ms)
        result = self._get("/v5/market/kline", params)
        tf = f"{interval}m" if interval.isdigit() else interval

        candles: list[Candle] = []
        for item in result.get("list", []):
            # [startTime, open, high, low, close, volume, turnover]
            try:
                candles.append(
                    Candle(
                        ts=int(item[0]),
                        symbol=symbol,
                        tf=tf,
                        open=float(item[1]),
                        high=float(item[2]),
                        low=float(item[3]),
                        close=float(item[4]),
                        volume=float(item[5]),
                    )
                )
            except (IndexError, TypeError, ValueError) as exc:
                logger.warning("skip malformed kline symbol=%s err=%s item=%r", symbol, exc, item)
        candles.sort(key=lambda c: c.ts)
        return candles

    def _ticker_item(self, symbol: str) -> dict[str, Any]:
        result = self._get("/v5/market/tickers", {"category": self.category, "symbol": symbol})
        items = result.get("list", [])
        if not items:
            raise RuntimeError(f"Bybit tickers returned no data for symbol={symbol}")
        return items[0]

    def get_ticker(self, symbol: str) -> Ticker:
        item = self._ticker_item(symbol)
        return Ticker(
            ts=self._now_ms(),
            symbol=symbol,
            bid=float(item.get("bid1Price") or 0.0),
            ask=float(item.get("ask1Price") or 0.0),
            last=float(item.get("lastPrice") or 0.0),
        )

    def get_open_interest(self, symbol: str) -> OpenInterest:
        item = self._ticker_item(symbol)
        return OpenInterest(ts=self._now_ms(), symbol=symbol, oi=float(item.get("openInterest") or 0.0))

    def get_funding(self, symbol: str) -> FundingRate:
        item = self._ticker_item(symbol)
        return FundingRate(ts=self._now_ms(), symbol=symbol, funding=float(item.get("fundingRate") or 0.0))

    def get_market_snapshot(self, symbol: str) -> tuple[Ticker, OpenInterest, FundingRate]:
        """Ticker + OI + funding from a single tickers call (one HTTP request)."""
        item = self._ticker_item(symbol)
        ts = self._now_ms()
        ticker = Ticker(
            ts=ts,
            symbol=symbol,
            bid=float(item.get("bid1Price") or 0.0),
            ask=float(item.get("ask1Price") or 0.0),
            last=float(item.get("lastPrice") or 0.0),
        )
        oi = OpenInterest(ts=ts, symbol=symbol, oi=float(item.get("openInterest") or 0.0))
        funding = FundingRate(ts=ts, symbol=symbol, funding=float(item.get("fundingRate") or 0.0))
        return ticker, oi, funding

    @staticmethod
    def _now_ms() -> int:
        from datetime import datetime, timezone

        return int(datetime.now(timezone.utc).timestamp() * 1000)
