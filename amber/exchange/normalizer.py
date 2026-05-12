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
