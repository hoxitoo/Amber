from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
from amber.common.logging import setup_logging
from amber.exchange.streams import BybitWSClient
from amber.storage.parquet_sink import ParquetSink


async def main() -> None:
    cfg = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    setup_logging(cfg.get("run", {}).get("log_level", "INFO"))

    symbols = cfg["exchange"]["bybit"]["symbols"]
    ws_url = cfg["exchange"]["bybit"]["ws_url"]
    sink = ParquetSink(Path(cfg["storage"]["raw_dir"]))

    topics = [f"kline.1.{s}" for s in symbols]

    async def handler(payload: dict) -> None:
        # Keep raw payload for later normalizer stage.
        topic = str(payload.get("topic", "unknown"))
        symbol = topic.split(".")[-1] if "." in topic else "unknown"
        sink.write_records(topic="ws_raw", symbol=symbol, records=[payload])

    client = BybitWSClient(ws_url=ws_url, topics=topics, handler=handler)
    await client.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
