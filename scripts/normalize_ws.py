from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
from amber.exchange.normalizer import BybitNormalizer, gap_fill
from amber.storage.parquet_sink import ParquetSink


if __name__ == "__main__":
    cfg = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    raw_root = Path(cfg["storage"]["raw_dir"])
    sink = ParquetSink(raw_root)

    normalizer = BybitNormalizer()
    step_ms = 60_000
    last_row_by_symbol = {}

    for file in sorted((raw_root / "ws_raw").glob("*/part-000.jsonl")):
        symbol = file.parent.name
        lines = [json.loads(x) for x in file.read_text(encoding="utf-8").splitlines() if x.strip()]
        out = []
        for payload in lines:
            candle = normalizer.candle_from_ws(payload)
            if candle is None:
                continue
            row = normalizer.to_normalized(candle)

            last = last_row_by_symbol.get(symbol)
            if last is not None and row.ts - last.ts > step_ms:
                out.extend([x.model_dump() for x in gap_fill(last, row.ts, step_ms)])

            out.append(row.model_dump())
            last_row_by_symbol[symbol] = row

        if out:
            sink.write_records(topic="normalized", symbol=symbol, records=out)
