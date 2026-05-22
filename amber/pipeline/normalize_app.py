from __future__ import annotations

import json
import logging
from pathlib import Path

from amber.common.config import ConfigLoader
from amber.common.logging import setup_logging
from amber.exchange.normalizer import BybitNormalizer, gap_fill
from amber.storage.parquet_sink import ParquetSink

logger = logging.getLogger(__name__)


<<<<<<< HEAD
def _iter_jsonl_payloads(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skip invalid ws json line file=%s line=%s", path, lineno)


=======
>>>>>>> origin/main
def main() -> None:
    cfg = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    setup_logging(cfg.get("run", {}).get("log_level", "INFO"))

    raw_root = Path(cfg["storage"]["raw_dir"])
    sink = ParquetSink(raw_root)

    normalizer = BybitNormalizer()
    step_ms = 60_000
    last_row_by_symbol = {}
    written = 0

    for file in sorted((raw_root / "ws_raw").glob("*/part-000.jsonl")):
        symbol = file.parent.name
<<<<<<< HEAD
        out = []
        for payload in _iter_jsonl_payloads(file):
=======
        lines = [json.loads(x) for x in file.read_text(encoding="utf-8").splitlines() if x.strip()]
        out = []
        for payload in lines:
>>>>>>> origin/main
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
            written += len(out)

    logger.info("normalize_app finished written=%s", written)


if __name__ == "__main__":
    main()
