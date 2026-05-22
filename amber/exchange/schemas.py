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
    is_synthetic: bool = Field(default=False)
