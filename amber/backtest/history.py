"""Long candle and funding history for slow-timeframe studies (4H / 1D).

Two public sources, no keys:

- `bybit`: the v5 REST API Amber already uses (`BybitPublicClient`). This is
  the venue the owner trades on; run it on the VPS.
- `binance`: the public archive at data.binance.vision (USD-M perpetuals,
  monthly zips from 2020). Reachable where Bybit's API is geo-blocked. At 4H
  the BTC and ETH perpetuals of the two venues track each other to a few
  basis points, so it stands in for Bybit history; the Bybit run is the check.

Bars come back oldest-first as dicts {ts, open, high, low, close} (ts = bar
open, ms UTC); funding as [(ts_ms, rate)].
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import logging
from pathlib import Path
import time
from typing import Any
import zipfile

logger = logging.getLogger(__name__)

BINANCE_ARCHIVE = "https://data.binance.vision/data/futures/um/monthly"
BYBIT_INTERVALS = {"4h": "240", "1d": "D"}
INTERVAL_MS = {"4h": 4 * 3_600_000, "1d": 86_400_000}


def _months(start: datetime, end: datetime) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _binance_csv(url: str, cache: Path) -> list[list[str]] | None:
    """One monthly archive, cached as its CSV. None if it does not exist."""
    if cache.exists():
        text = cache.read_text(encoding="utf-8")
    else:
        import httpx

        resp = httpx.get(url, timeout=30.0, follow_redirects=True)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            text = z.read(z.namelist()[0]).decode("utf-8")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    # Archives from 2022 on carry a header row; earlier ones do not.
    if rows and rows[0] and not rows[0][0].strip().lstrip("-").isdigit():
        rows = rows[1:]
    return rows


def binance_history(
    symbol: str,
    interval: str,
    cache_dir: Path,
    *,
    start: datetime = datetime(2020, 1, 1, tzinfo=timezone.utc),
    end: datetime | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[int, float]]]:
    """Complete months only: the current month has no monthly archive yet."""
    end = end or datetime.now(timezone.utc)
    bars: dict[int, dict[str, Any]] = {}
    funding: dict[int, float] = {}
    for y, m in _months(start, end):
        tag = f"{y:04d}-{m:02d}"
        rows = _binance_csv(f"{BINANCE_ARCHIVE}/klines/{symbol}/{interval}/{symbol}-{interval}-{tag}.zip",
                            cache_dir / f"{symbol}-{interval}-{tag}.csv")
        for r in rows or []:
            try:
                bars[int(r[0])] = {"ts": int(r[0]), "open": float(r[1]), "high": float(r[2]),
                                   "low": float(r[3]), "close": float(r[4])}
            except (IndexError, ValueError):
                continue
        frows = _binance_csv(f"{BINANCE_ARCHIVE}/fundingRate/{symbol}/{symbol}-fundingRate-{tag}.zip",
                             cache_dir / f"{symbol}-funding-{tag}.csv")
        for r in frows or []:
            try:
                funding[int(r[0])] = float(r[2])
            except (IndexError, ValueError):
                continue
    return [bars[k] for k in sorted(bars)], sorted(funding.items())


def bybit_history(
    symbol: str,
    interval: str,
    *,
    start_ms: int = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000),
    client: Any | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[int, float]]]:
    """Paged backwards from now; Bybit returns at most 1000 bars / 200 rates a call."""
    from amber.exchange.bybit_public import BybitPublicClient

    own = client is None
    client = client or BybitPublicClient()
    try:
        bars: dict[int, dict[str, Any]] = {}
        end = int(time.time() * 1000)
        while end > start_ms:
            page = client.get_klines(symbol, interval=BYBIT_INTERVALS[interval], limit=1000, end_ms=end)
            if not page:
                break
            for c in page:
                bars[c.ts] = {"ts": c.ts, "open": c.open, "high": c.high, "low": c.low, "close": c.close}
            oldest = min(c.ts for c in page)
            if oldest >= end:
                break
            end = oldest - 1
        funding: dict[int, float] = {}
        end = int(time.time() * 1000)
        while end > start_ms:
            res = client._get("/v5/market/funding/history",
                              {"category": "linear", "symbol": symbol, "endTime": end, "limit": 200})
            items = res.get("list", [])
            if not items:
                break
            stamps = []
            for it in items:
                try:
                    ts = int(it["fundingRateTimestamp"])
                    funding[ts] = float(it["fundingRate"])
                    stamps.append(ts)
                except (KeyError, TypeError, ValueError):
                    continue
            if not stamps or min(stamps) >= end:
                break
            end = min(stamps) - 1
    finally:
        if own:
            client.close()
    # The newest bar is still forming.
    now = int(time.time() * 1000)
    out = [bars[k] for k in sorted(bars) if k >= start_ms and k + INTERVAL_MS[interval] <= now]
    return out, sorted(funding.items())


def load_history(source: str, symbol: str, interval: str, cache_dir: Path):
    if source == "binance":
        return binance_history(symbol, interval, cache_dir / "binance")
    if source == "bybit":
        return bybit_history(symbol, interval)
    raise ValueError(f"unknown source {source!r}")
