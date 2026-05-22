from __future__ import annotations

from dataclasses import dataclass, field

from amber.exchange.schemas import Candle, FundingRate, NormalizedRow, OpenInterest, Ticker


@dataclass(slots=True)
class MarketStateCache:
    tickers: dict[str, Ticker] = field(default_factory=dict)
    oi: dict[str, OpenInterest] = field(default_factory=dict)
    funding: dict[str, FundingRate] = field(default_factory=dict)


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
    def candle_from_ws(payload: dict) -> Candle | None:
        """Parse Bybit kline payload item into Candle contract."""
        try:
            topic = str(payload.get("topic", ""))
            if not topic.startswith("kline"):
                return None
            data = payload.get("data", [])
            if not data:
                return None
            item = data[0]
            tf = str(item.get("interval", "1"))
            symbol = str(item["symbol"])
            ts = int(item.get("start", 0))
            return Candle(
                ts=ts,
                symbol=symbol,
                tf=f"{tf}m" if tf.isdigit() else tf,
                open=float(item["open"]),
                high=float(item["high"]),
                low=float(item["low"]),
                close=float(item["close"]),
                volume=float(item.get("volume", 0.0)),
            )
        except Exception:
            return None


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
