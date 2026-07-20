from __future__ import annotations

import logging
from dataclasses import dataclass, field

from amber.exchange.schemas import Candle, FundingRate, NormalizedRow, OpenInterest, Ticker

logger = logging.getLogger(__name__)

_TF_UNIT_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}


def step_ms_for_tf(tf: str) -> int:
    """Candle period in ms for a timeframe like '1m', '5m', '1h'. Defaults to 1m."""
    tf = tf.strip().lower()
    if len(tf) >= 2 and tf[:-1].isdigit() and tf[-1] in _TF_UNIT_MS:
        return int(tf[:-1]) * _TF_UNIT_MS[tf[-1]]
    if tf.isdigit():
        return int(tf) * 60_000
    logger.warning("unknown timeframe %r, assuming 1m", tf)
    return 60_000


@dataclass(slots=True)
class MarketStateCache:
    tickers: dict[str, Ticker] = field(default_factory=dict)
    oi: dict[str, OpenInterest] = field(default_factory=dict)
    funding: dict[str, FundingRate] = field(default_factory=dict)


def _opt_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class BybitNormalizer:
    def __init__(self) -> None:
        self.cache = MarketStateCache()

    def update_ticker(self, ticker: Ticker) -> None:
        self.cache.tickers[ticker.symbol] = ticker

    def update_oi(self, oi: OpenInterest) -> None:
        self.cache.oi[oi.symbol] = oi

    def update_funding(self, funding: FundingRate) -> None:
        self.cache.funding[funding.symbol] = funding

    def to_normalized(self, candle: Candle, is_synthetic: bool = False) -> NormalizedRow:
        t = self.cache.tickers.get(candle.symbol)
        oi = self.cache.oi.get(candle.symbol)
        f = self.cache.funding.get(candle.symbol)

        return NormalizedRow(
            ts=candle.ts,
            symbol=candle.symbol,
            tf=candle.tf,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            bid=t.bid if t else candle.close,
            ask=t.ask if t else candle.close,
            oi=oi.oi if oi else 0.0,
            funding=f.funding if f else 0.0,
            is_synthetic=is_synthetic,
        )

    @staticmethod
    def symbol_from_topic(topic: str) -> str | None:
        """Extract symbol from a Bybit v5 topic like 'kline.1.BTCUSDT' or 'tickers.BTCUSDT'."""
        parts = topic.split(".")
        if len(parts) < 2 or not parts[-1]:
            return None
        return parts[-1]

    @staticmethod
    def candles_from_ws(payload: dict, *, confirmed_only: bool = True) -> list[Candle]:
        """Parse a Bybit v5 kline WS payload into confirmed Candle contracts.

        Bybit does not include the symbol inside data items — it lives in the topic.
        Unconfirmed (in-progress) candle updates are skipped by default because the
        stream re-sends the same candle many times before it closes.
        """
        topic = str(payload.get("topic", ""))
        if not topic.startswith("kline."):
            return []
        symbol = BybitNormalizer.symbol_from_topic(topic)
        if symbol is None:
            logger.warning("kline payload without symbol in topic=%r", topic)
            return []

        data = payload.get("data")
        if not isinstance(data, list):
            return []

        out: list[Candle] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if confirmed_only and not bool(item.get("confirm", False)):
                continue
            try:
                tf = str(item.get("interval", "1"))
                out.append(
                    Candle(
                        ts=int(item["start"]),
                        symbol=symbol,
                        tf=f"{tf}m" if tf.isdigit() else tf,
                        open=float(item["open"]),
                        high=float(item["high"]),
                        low=float(item["low"]),
                        close=float(item["close"]),
                        volume=float(item.get("volume", 0.0)),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("skip malformed kline item topic=%r err=%s item=%r", topic, exc, item)
        return out

    @staticmethod
    def candle_from_ws(payload: dict, *, confirmed_only: bool = True) -> Candle | None:
        """Parse a Bybit kline payload into the last confirmed Candle (or None)."""
        candles = BybitNormalizer.candles_from_ws(payload, confirmed_only=confirmed_only)
        return candles[-1] if candles else None

    def update_from_ws_ticker(self, payload: dict) -> bool:
        """Merge a Bybit v5 `tickers.*` WS payload (snapshot or delta) into the cache.

        Delta messages carry only changed fields, so absent fields keep prior values.
        Returns True when the payload was a ticker payload (even if nothing changed).
        """
        topic = str(payload.get("topic", ""))
        if not topic.startswith("tickers."):
            return False
        symbol = self.symbol_from_topic(topic)
        data = payload.get("data")
        if symbol is None or not isinstance(data, dict):
            return True

        ts = int(_opt_float(payload.get("ts")) or 0)

        prev = self.cache.tickers.get(symbol)
        bid = _opt_float(data.get("bid1Price"))
        ask = _opt_float(data.get("ask1Price"))
        last = _opt_float(data.get("lastPrice"))
        merged_bid = bid if bid is not None else (prev.bid if prev else None)
        merged_ask = ask if ask is not None else (prev.ask if prev else None)
        merged_last = last if last is not None else (prev.last if prev else merged_bid)
        if merged_bid is not None and merged_ask is not None:
            self.cache.tickers[symbol] = Ticker(
                ts=ts or (prev.ts if prev else 0),
                symbol=symbol,
                bid=merged_bid,
                ask=merged_ask,
                last=merged_last if merged_last is not None else merged_bid,
            )

        oi = _opt_float(data.get("openInterest"))
        if oi is not None:
            self.cache.oi[symbol] = OpenInterest(ts=ts, symbol=symbol, oi=oi)

        funding = _opt_float(data.get("fundingRate"))
        if funding is not None:
            self.cache.funding[symbol] = FundingRate(ts=ts, symbol=symbol, funding=funding)
        return True


def gap_fill(last: NormalizedRow, next_ts: int, step_ms: int) -> list[NormalizedRow]:
    """Forward-fill missing candles between `last.ts` and `next_ts`.

    Generated rows are marked as synthetic and have zero volume.
    """
    out: list[NormalizedRow] = []
    cur = last.ts + step_ms
    while cur < next_ts:
        out.append(
            NormalizedRow(
                ts=cur,
                symbol=last.symbol,
                tf=last.tf,
                open=last.close,
                high=last.close,
                low=last.close,
                close=last.close,
                volume=0.0,
                bid=last.bid,
                ask=last.ask,
                oi=last.oi,
                funding=last.funding,
                is_synthetic=True,
            )
        )
        cur += step_ms
    return out
