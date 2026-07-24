from __future__ import annotations

import asyncio
from collections import defaultdict
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


async def main() -> None:
    cfg = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    setup_logging(cfg.get("run", {}).get("log_level", "INFO"))

    symbols = cfg["exchange"]["bybit"]["symbols"]
    ws_url = cfg["exchange"]["bybit"]["ws_url"]
    sink = ParquetSink(Path(cfg["storage"]["raw_dir"]))

    # Klines drive the candle series; tickers carry bid/ask, open interest and
    # funding rate; publicTrade carries taker aggressor flow (CVD/imbalance).
    topics = (
        [f"kline.1.{s}" for s in symbols]
        + [f"tickers.{s}" for s in symbols]
        + [f"publicTrade.{s}" for s in symbols]
    )

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
        for symbol, records in by_symbol.items():
            sink.write_records(topic="ws_raw", symbol=symbol, records=records)

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
