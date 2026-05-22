from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import sqrt
from statistics import mean
from typing import Any


@dataclass(slots=True)
class SymbolWindow:
    close: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    volume: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    oi: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    funding: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    bid: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    ask: deque[float] = field(default_factory=lambda: deque(maxlen=120))


class FeatureEngine:
    """Unified feature logic for offline/live usage.

    - `update(row)` for streaming state updates
    - `latest(symbol)` to retrieve current feature vector
    - `transform_rows(rows)` for offline recomputation using same formulas
    """

    def __init__(self) -> None:
        self.state: dict[str, SymbolWindow] = defaultdict(SymbolWindow)

    def update(self, row: dict[str, Any]) -> dict[str, Any]:
        symbol = str(row["symbol"])
        w = self.state[symbol]

        close = float(row["close"])
        volume = float(row.get("volume", 0.0))
        oi = float(row.get("oi", 0.0))
        funding = float(row.get("funding", 0.0))
        bid = float(row.get("bid", close))
        ask = float(row.get("ask", close))

        w.close.append(close)
        w.volume.append(volume)
        w.oi.append(oi)
        w.funding.append(funding)
        w.bid.append(bid)
        w.ask.append(ask)

        return self._build_features(symbol=symbol, ts=row["ts"])

    def latest(self, symbol: str) -> dict[str, Any] | None:
        if symbol not in self.state or not self.state[symbol].close:
            return None
        return self._build_features(symbol=symbol, ts=0)

    def transform_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(self.update(row))
        return out

    def _build_features(self, symbol: str, ts: int) -> dict[str, Any]:
        w = self.state[symbol]

        def _ret(n: int) -> float:
            if len(w.close) <= n:
                return 0.0
            prev = w.close[-(n + 1)]
            cur = w.close[-1]
            return 0.0 if prev == 0 else (cur / prev) - 1.0

        def _z(values: deque[float], lookback: int) -> float:
            if len(values) < max(3, lookback):
                return 0.0
            sample = list(values)[-lookback:]
            m = mean(sample)
            var = mean([(x - m) ** 2 for x in sample])
            std = sqrt(var) if var > 0 else 0.0
            return 0.0 if std == 0 else (sample[-1] - m) / std

        mid_price = (w.bid[-1] + w.ask[-1]) / 2.0 if w.bid and w.ask else w.close[-1]
        spread_bps = 0.0 if mid_price == 0 else ((w.ask[-1] - w.bid[-1]) / mid_price) * 10_000
        return {
            "symbol": symbol,
            "ts": ts,
            "ret_1": _ret(1),
            "ret_5": _ret(5),
            "ret_20": _ret(20),
            "vol_z_20": _z(w.volume, 20),
            "oi_z_20": _z(w.oi, 20),
            "funding_z_20": _z(w.funding, 20),
            "mid_price": mid_price,
            "bid": w.bid[-1] if w.bid else mid_price,
            "ask": w.ask[-1] if w.ask else mid_price,
            "spread_bps": spread_bps,
            "obs": len(w.close),
        }
