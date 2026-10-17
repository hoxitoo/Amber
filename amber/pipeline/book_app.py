"""Order book + sub-minute flow collector (roadmap D2), a service of its own.

Every 1m feature Amber has says "a move is under way", and none says which
way (CLAUDE.md section 10). The order book and the taker flow inside the
minute are the public data not yet looked at for direction. They cannot be
fetched for the past, so they have to be collected before they can be tested.

Runs as `amber-book-collector`, separate from `amber-ws-collector`: its own
socket, process and systemd limits, so a fault here cannot stop the candles.
Every `interval_sec` it writes one row per symbol (amber/exchange/orderbook.py)
to `raw/book/<SYMBOL>/<YYYY-MM-DD>.csv`; a finished day is gzipped, days
older than `retention_days` are deleted, and nothing is written while the
disk has less than `LOW_DISK_BYTES` free. Every `STATS_SEC` it logs what it
costs: messages per second, CPU share, memory, bytes written.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import gzip
import logging
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any

from amber.common.retention import free_bytes
from amber.exchange.orderbook import COLUMNS, Book, TradeBuckets, format_row

logger = logging.getLogger(__name__)

LOW_DISK_BYTES = 2_000_000_000
STATS_SEC = 600
STALE_MS = 10_000
SAMPLE_LAG_SEC = 0.5  # let the interval's last trades arrive before writing it


def book_topics(symbols: list[str], depth: int, trades: bool) -> list[str]:
    topics = [f"orderbook.{depth}.{s}" for s in symbols]
    if trades:
        topics += [f"publicTrade.{s}" for s in symbols]
    return topics


def _day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _rss_mb() -> float:
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
    except (OSError, ValueError, IndexError):  # pragma: no cover - non-Linux
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class BookWriter:
    """Appends rows to one CSV per symbol per UTC day; gzips and expires old days."""

    def __init__(self, root: Path, retention_days: int) -> None:
        self.root = Path(root)
        self.retention_days = int(retention_days)
        self.day: str | None = None
        self.handles: dict[str, Any] = {}
        self.bytes_written = 0
        self.disk_low = False

    def write(self, ts_ms: int, lines: dict[str, str]) -> None:
        day = _day(ts_ms)
        if day != self.day:
            # A new day, or the first write after a start: earlier days are
            # finished whether or not this process saw them end.
            self.close()
            self.housekeep(day)
            self.day = day
        if free_bytes(self.root) < LOW_DISK_BYTES:
            if not self.disk_low:
                logger.warning("book: under %.1f GB free on disk, not writing until it recovers",
                               LOW_DISK_BYTES / 1e9)
            self.disk_low = True
            return
        self.disk_low = False
        for symbol, line in lines.items():
            fh = self.handles.get(symbol)
            if fh is None:
                path = self.root / symbol / f"{day}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                new = not path.exists() or path.stat().st_size == 0
                fh = path.open("a", encoding="utf-8")
                if new:
                    fh.write(",".join(COLUMNS) + "\n")
                self.handles[symbol] = fh
            fh.write(line + "\n")
            self.bytes_written += len(line) + 1
        for fh in self.handles.values():
            fh.flush()

    def close(self) -> None:
        for fh in self.handles.values():
            try:
                fh.close()
            except OSError:
                pass
        self.handles.clear()

    def housekeep(self, today: str) -> None:
        """Gzip every finished day; delete days past retention."""
        cutoff = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=self.retention_days)).strftime("%Y-%m-%d")
        if not self.root.exists():
            return
        for path in sorted(self.root.glob("*/*.csv*")):
            day = path.name.split(".")[0]
            try:
                if day < cutoff:
                    path.unlink()
                elif day < today and path.suffix == ".csv":
                    gz = path.with_name(path.name + ".gz")
                    # Append, never overwrite: should that day's archive already
                    # exist (a clock stepped back over midnight), its rows stay.
                    # gzip readers read concatenated members as one file.
                    with path.open("rb") as src, gzip.open(gz, "ab", compresslevel=6) as dst:
                        shutil.copyfileobj(src, dst)
                    path.unlink()
            except OSError as exc:
                logger.warning("book: housekeeping failed on %s: %s", path, exc)


class BookCollector:
    def __init__(self, symbols: list[str], root: Path, *, interval_sec: int = 5,
                 retention_days: int = 45) -> None:
        self.symbols = list(symbols)
        self.interval_ms = int(interval_sec) * 1000
        self.books = {s: Book() for s in self.symbols}
        self.flow = TradeBuckets(self.interval_ms)
        self.writer = BookWriter(Path(root), retention_days)
        self.msgs: dict[str, int] = {"orderbook": 0, "publicTrade": 0, "other": 0}
        self.rows = 0
        self.stale = 0
        self.last_msg_ms = 0  # local clock: immune to skew against Bybit's
        self.last_end = 0

    async def handle(self, payload: dict) -> None:
        self.last_msg_ms = int(time.time() * 1000)
        topic = str(payload.get("topic", ""))
        if topic.startswith("orderbook."):
            self.msgs["orderbook"] += 1
            symbol = topic.rsplit(".", 1)[-1]
            book = self.books.get(symbol)
            data = payload.get("data")
            if book is not None and isinstance(data, dict):
                book.apply(str(payload.get("type", "")), data, int(payload.get("ts", 0) or 0))
        elif topic.startswith("publicTrade."):
            self.msgs["publicTrade"] += 1
            symbol = topic.rsplit(".", 1)[-1]
            for trade in payload.get("data") or ():
                if isinstance(trade, dict):
                    self.flow.add(symbol, trade)
        else:
            self.msgs["other"] += 1

    def sample(self, end_ms: int, now_ms: int) -> dict[str, str]:
        """Rows for the interval ending at `end_ms`, the book as it stands now.

        A book is only as current as the socket: Bybit sends a delta only when
        a level changes, so a quiet coin's book can be minutes old and still
        right, but if nothing at all has arrived for STALE_MS the connection is
        down and every book is the one from before the drop. Those rows keep
        their flow and leave the book empty.
        """
        if end_ms <= self.last_end:
            return {}  # the clock stepped back: this interval is written already
        self.last_end = end_ms
        start = end_ms - self.interval_ms
        socket_dead = now_ms - self.last_msg_ms > STALE_MS
        lines = {}
        for s in self.symbols:
            book = self.books[s]
            feats = None if socket_dead else book.features()
            if socket_dead and book.ready:
                self.stale += 1
            age = now_ms - book.updated_ms if book.updated_ms else None
            lines[s] = format_row(end_ms, feats, self.flow.take(s, start), age)
        self.rows += len(lines)
        return lines

    async def sampler(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            now = time.time()
            nxt = (int(now * 1000) // self.interval_ms + 1) * self.interval_ms
            await asyncio.sleep(max(0.0, nxt / 1000 - now + SAMPLE_LAG_SEC))
            lines = self.sample(nxt, int(time.time() * 1000))
            if lines:
                await loop.run_in_executor(None, self.writer.write, nxt, lines)

    async def stats(self) -> None:
        last_wall, last_cpu = time.time(), time.process_time()
        last = dict(self.msgs)
        last_bytes = 0
        while True:
            await asyncio.sleep(STATS_SEC)
            wall, cpu = time.time(), time.process_time()
            dt = max(1e-9, wall - last_wall)
            rates = {k: (v - last.get(k, 0)) / dt for k, v in self.msgs.items()}
            mb = (self.writer.bytes_written - last_bytes) / 1e6
            logger.info(
                "book: %.0f book msg/s, %.0f trade msg/s, CPU %.0f%% of one core, RSS %.0f MB, "
                "%s rows, %.1f MB written in %.0f min (~%.0f MB/day), stale book rows %s, crossed %s, "
                "late trades %s",
                rates["orderbook"], rates["publicTrade"], (cpu - last_cpu) / dt * 100, _rss_mb(),
                self.rows, mb, dt / 60, mb / dt * 86400, self.stale,
                sum(b.crossed for b in self.books.values()), self.flow.late,
            )
            last_wall, last_cpu, last, last_bytes = wall, cpu, dict(self.msgs), self.writer.bytes_written


async def run(config: dict, *, ws_url: str | None = None) -> None:
    from amber.exchange.streams import BybitWSClient

    bybit = config["exchange"]["bybit"]
    book_cfg = config.get("book", {}) if isinstance(config.get("book"), dict) else {}
    symbols = list(bybit["symbols"])
    root = Path(config["storage"]["raw_dir"]) / "book"
    collector = BookCollector(symbols, root, interval_sec=int(book_cfg.get("interval_sec", 5)),
                              retention_days=int(book_cfg.get("retention_days", 45)))
    topics = book_topics(symbols, int(book_cfg.get("depth", 200)), bool(book_cfg.get("trades", True)))
    client = BybitWSClient(ws_url=ws_url or bybit["ws_url"], topics=topics, handler=collector.handle)
    logger.info("book collector: %s symbols, %s topics, every %ss, into %s",
                len(symbols), len(topics), book_cfg.get("interval_sec", 5), root)
    try:
        await asyncio.gather(client.run_forever(), collector.sampler(), collector.stats())
    finally:
        collector.writer.close()
