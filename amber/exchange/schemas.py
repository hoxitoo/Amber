from __future__ import annotations

from pydantic import BaseModel, Field


class Candle(BaseModel):
    ts: int
    symbol: str
    tf: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class Ticker(BaseModel):
    ts: int
    symbol: str
    bid: float
    ask: float
    last: float


class OpenInterest(BaseModel):
    ts: int
    symbol: str
    oi: float


class FundingRate(BaseModel):
    ts: int
    symbol: str
    funding: float


class NormalizedRow(BaseModel):
    ts: int
    symbol: str
    tf: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float
    ask: float
    oi: float
    funding: float
    # Taker order flow for the candle's minute (from publicTrade); 0 when no
    # trade stream is available (e.g. REST backfill or a quiet minute).
    buy_volume: float = Field(default=0.0)
    sell_volume: float = Field(default=0.0)
    trade_count: int = Field(default=0)
    is_synthetic: bool = Field(default=False)
