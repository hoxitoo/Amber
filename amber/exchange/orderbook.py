"""Local order book from Bybit's public `orderbook.{depth}.{symbol}` stream,
and the compact per-interval row the book collector stores (roadmap D2).

Bybit sends a snapshot, then deltas: a level with size "0" is removed, any
other size replaces the level. A message with update id `u == 1` is a fresh
snapshot after a service restart and must overwrite the book. Deltas that
arrive before the first snapshot are ignored, because applied to an empty book
they would describe a book that does not exist.

The stored row is not the book itself (200 levels x 2 sides every 100 ms
would be gigabytes a day) but what a direction study needs from it: mid,
spread, top-of-book size and the resting notional within 5 / 10 / 25 / 50 /
100 bps of the mid on each side, how far the received depth reaches, and the
taker flow of the interval.
"""

from __future__ import annotations

from typing import Any

BANDS_BPS = (5, 10, 25, 50, 100)

COLUMNS = (
    ["ts", "mid", "spread_bps", "bid1_usd", "ask1_usd"]
    + [f"{side}{b}_usd" for b in BANDS_BPS for side in ("bid", "ask")]
    + ["bid_cov_bps", "ask_cov_bps", "buy_usd", "sell_usd", "n_buy", "n_sell", "max_trade_usd", "book_age_ms"]
)


class Book:
    __slots__ = ("bids", "asks", "ready", "updated_ms", "crossed")

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.ready = False
        self.updated_ms = 0
        self.crossed = 0

    def apply(self, msg_type: str, data: dict[str, Any], ts_ms: int) -> None:
        if msg_type == "snapshot" or int(data.get("u", 0) or 0) == 1:
            self.bids.clear()
            self.asks.clear()
            self.ready = True
        elif not self.ready:
            return
        for side, levels in ((self.bids, data.get("b") or ()), (self.asks, data.get("a") or ())):
            for level in levels:
                try:
                    price, size = float(level[0]), float(level[1])
                except (IndexError, TypeError, ValueError):
                    continue
                if size <= 0:
                    side.pop(price, None)
                else:
                    side[price] = size
        self.updated_ms = int(ts_ms)

    def reset(self) -> None:
        """After a reconnect: wait for the new snapshot instead of trusting the old book."""
        self.bids.clear()
        self.asks.clear()
        self.ready = False

    def features(self) -> dict[str, float] | None:
        """None when there is no usable book (not ready, one side empty, crossed)."""
        if not self.ready or not self.bids or not self.asks:
            return None
        best_bid, best_ask = max(self.bids), min(self.asks)
        if best_bid >= best_ask:
            self.crossed += 1
            return None
        mid = (best_bid + best_ask) / 2.0
        out: dict[str, float] = {
            "mid": mid,
            "spread_bps": (best_ask - best_bid) / mid * 1e4,
            "bid1_usd": best_bid * self.bids[best_bid],
            "ask1_usd": best_ask * self.asks[best_ask],
            "bid_cov_bps": (mid - min(self.bids)) / mid * 1e4,
            "ask_cov_bps": (max(self.asks) - mid) / mid * 1e4,
        }
        for b in BANDS_BPS:
            lo, hi = mid * (1 - b / 1e4), mid * (1 + b / 1e4)
            out[f"bid{b}_usd"] = sum(p * s for p, s in self.bids.items() if p >= lo)
            out[f"ask{b}_usd"] = sum(p * s for p, s in self.asks.items() if p <= hi)
        return out


class TradeBuckets:
    """Taker flow per symbol per interval, bucketed by the trade's own time."""

    def __init__(self, interval_ms: int) -> None:
        self.interval_ms = int(interval_ms)
        self.buckets: dict[str, dict[int, list[float]]] = {}
        self.emitted_until: dict[str, int] = {}
        self.late = 0

    def add(self, symbol: str, trade: dict[str, Any]) -> None:
        try:
            t = int(trade["T"])
            notional = float(trade["v"]) * float(trade["p"])
            buy = str(trade.get("S", "")).lower() == "buy"
        except (KeyError, TypeError, ValueError):
            return
        start = t - t % self.interval_ms
        if start < self.emitted_until.get(symbol, -1):
            self.late += 1  # its interval was already written
            return
        b = self.buckets.setdefault(symbol, {}).setdefault(start, [0.0, 0.0, 0, 0, 0.0])
        if buy:
            b[0] += notional
            b[2] += 1
        else:
            b[1] += notional
            b[3] += 1
        b[4] = max(b[4], notional)

    def take(self, symbol: str, start: int) -> list[float]:
        """Flow of [start, start + interval); later intervals stay buffered."""
        mine = self.buckets.get(symbol, {})
        for k in [k for k in mine if k < start]:
            self.late += int(sum(mine.pop(k)[2:4]))  # an interval the sampler skipped
        self.emitted_until[symbol] = start + self.interval_ms
        return mine.pop(start, [0.0, 0.0, 0, 0, 0.0])


def _num(v: float | None, fmt: str) -> str:
    return "" if v is None else format(v, fmt)


def format_row(ts: int, feats: dict[str, float] | None, flow: list[float], age_ms: int | None) -> str:
    f = feats or {}
    cells = [str(ts), _num(f.get("mid"), ".10g"), _num(f.get("spread_bps"), ".3f"),
             _num(f.get("bid1_usd"), ".0f"), _num(f.get("ask1_usd"), ".0f")]
    for b in BANDS_BPS:
        cells += [_num(f.get(f"bid{b}_usd"), ".0f"), _num(f.get(f"ask{b}_usd"), ".0f")]
    cells += [_num(f.get("bid_cov_bps"), ".1f"), _num(f.get("ask_cov_bps"), ".1f"),
              format(flow[0], ".0f"), format(flow[1], ".0f"), str(int(flow[2])), str(int(flow[3])),
              format(flow[4], ".0f"), "" if age_ms is None or not feats else str(int(age_ms))]
    return ",".join(cells)


def parse_row(line: str) -> dict[str, float | None] | None:
    """One stored row back to numbers; None for a torn or foreign line."""
    cells = line.rstrip("\n").split(",")
    if len(cells) != len(COLUMNS) or not cells[0].isdigit():
        return None
    out: dict[str, float | None] = {}
    for name, cell in zip(COLUMNS, cells):
        try:
            out[name] = float(cell) if cell != "" else None
        except ValueError:
            return None
    return out
