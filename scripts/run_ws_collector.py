from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
from amber.common.logging import setup_logging
from amber.exchange.streams import BybitWSClient
from amber.storage.parquet_sink import ParquetSink

logger = logging.getLogger(__name__)

FLUSH_INTERVAL_SEC = 1.0
FLUSH_BATCH = 500


# Bybit caps the subscribe `args` array at 21,000 characters per public
# connection (v5 docs, "Connect"). Futures have no per-request args count limit.
WS_ARGS_CHAR_LIMIT = 21_000


def build_topics(symbols: list[str], bybit_cfg: dict) -> list[str]:
    """Public stream topics to subscribe, in subscription order.

    Klines drive the candle series; tickers carry bid/ask, open interest and
    funding; publicTrade carries taker aggressor flow (CVD/imbalance) and is
    high-volume — disable it via `collect_trades: false` if disk is tight.

    Forced liquidations need no API key either. They are sparse — most minutes
    carry none — so the disk cost is small next to publicTrade. A cascade is the
    one event in this feed that can precede a move rather than describe it: the
    first liquidations force market orders, which move price, which trigger the
    next. At 27 symbols with all four streams the args run to 2,453 characters.
    """
    topics = [f"kline.1.{s}" for s in symbols] + [f"tickers.{s}" for s in symbols]
    if bool(bybit_cfg.get("collect_trades", True)):
        topics += [f"publicTrade.{s}" for s in symbols]
    if bool(bybit_cfg.get("collect_liquidations", True)):
        topics += [f"allLiquidation.{s}" for s in symbols]
    return topics


async def main() -> None:
    cfg = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    setup_logging(cfg.get("run", {}).get("log_level", "INFO"))

    symbols = cfg["exchange"]["bybit"]["symbols"]
    ws_url = cfg["exchange"]["bybit"]["ws_url"]
    sink = ParquetSink(Path(cfg["storage"]["raw_dir"]))

    topics = build_topics(symbols, cfg["exchange"]["bybit"])

    # Disk writes are batched off the WS read loop (audit A4): the handler only
    # enqueues, a writer task flushes by size/interval, so bursty markets can't
    # backpressure the socket with per-message fsyncs.
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=50_000)

    async def handler(payload: dict) -> None:
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            logger.warning("ws buffer full; dropping payload topic=%s", payload.get("topic"))

    def _flush(buffer: list[dict]) -> None:
        by_symbol: dict[str, list[dict]] = defaultdict(list)
        for payload in buffer:
            topic = str(payload.get("topic", "unknown"))
            symbol = topic.split(".")[-1] if "." in topic else "unknown"
            by_symbol[symbol].append(payload)
        # Rotate raw files hourly so the normalizer can delete consumed ones and
        # ws_raw doesn't grow without bound (publicTrade is high-volume).
        part = "part-" + datetime.now(timezone.utc).strftime("%Y%m%d%H")
        for symbol, records in by_symbol.items():
            sink.write_records(topic="ws_raw", symbol=symbol, records=records, part=part)

    async def writer() -> None:
        buffer: list[dict] = []
        loop = asyncio.get_running_loop()
        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=FLUSH_INTERVAL_SEC)
                buffer.append(payload)
            except asyncio.TimeoutError:
                pass
            if buffer and (len(buffer) >= FLUSH_BATCH or queue.empty()):
                batch, buffer = buffer, []
                await loop.run_in_executor(None, _flush, batch)

    client = BybitWSClient(ws_url=ws_url, topics=topics, handler=handler)
    await asyncio.gather(client.run_forever(), writer())


if __name__ == "__main__":
    asyncio.run(main())
