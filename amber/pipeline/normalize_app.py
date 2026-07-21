from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from amber.common.config import ConfigLoader
from amber.common.locks import AlreadyRunning, SingleInstanceLock
from amber.common.logging import setup_logging
from amber.exchange.normalizer import BybitNormalizer, gap_fill, step_ms_for_tf
from amber.storage.parquet_sink import ParquetSink
from amber.storage.state_store import StateStore

logger = logging.getLogger(__name__)

OFFSETS_STATE_KEY = "normalize_offsets"
WATERMARK_STATE_KEY = "normalized_watermark"


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


def _read_new_payloads(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    """Read JSONL payloads starting at byte `offset`; return (payloads, new_offset).

    Invalid lines are skipped with a warning. If the file shrank below the stored
    offset (rotation/truncation), reading restarts from the beginning.
    """
    size = path.stat().st_size
    if offset > size:
        logger.warning("offset %s beyond file size %s, resetting file=%s", offset, size, path)
        offset = 0

    payloads: list[dict[str, Any]] = []
    with path.open("rb") as fh:
        fh.seek(offset)
        for raw in fh:
            if not raw.endswith(b"\n"):
                # partial trailing line still being written — leave it for the next run
                break
            offset += len(raw)
            line = raw.strip()
            if not line:
                continue
            try:
                payloads.append(json.loads(line.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning("skip invalid ws json line file=%s", path)
    return payloads, offset


def normalize_ws_raw(raw_root: Path, state: StateStore) -> int:
    """Incrementally normalize raw WS payloads into `normalized/<symbol>/`.

    Idempotent across runs: per-file byte offsets and per-symbol last-candle
    timestamps are persisted in the state store, so reprocessing never duplicates
    rows. Ticker payloads update the market-state cache; only confirmed klines
    become normalized rows.
    """
    sink = ParquetSink(raw_root)
    normalizer = BybitNormalizer()

    offsets: dict[str, int] = dict(state.get(OFFSETS_STATE_KEY))
    watermark = state.get(WATERMARK_STATE_KEY)
    last_ts: dict[str, int] = {k: int(v) for k, v in watermark.get("last_ts", {}).items()}
    last_row_by_symbol: dict[str, Any] = {}

    written = 0
    for file in sorted((raw_root / "ws_raw").glob("*/part-*.jsonl")):
        symbol = file.parent.name
        key = f"{symbol}/{file.name}"
        payloads, new_offset = _read_new_payloads(file, int(offsets.get(key, 0)))
        offsets[key] = new_offset

        out: list[dict[str, Any]] = []
        for payload in payloads:
            if normalizer.update_from_ws_ticker(payload):
                continue
            for candle in normalizer.candles_from_ws(payload):
                if candle.ts <= last_ts.get(symbol, -1):
                    continue
                row = normalizer.to_normalized(candle)
                last = last_row_by_symbol.get(symbol)
                step_ms = step_ms_for_tf(row.tf)
                if last is not None and row.ts - last.ts > step_ms:
                    out.extend([x.model_dump() for x in gap_fill(last, row.ts, step_ms)])
                out.append(row.model_dump())
                last_row_by_symbol[symbol] = row
                last_ts[symbol] = candle.ts

        if out:
            sink.write_records(topic="normalized", symbol=symbol, records=out)
            written += len(out)

    state.set(OFFSETS_STATE_KEY, offsets)
    state.set(WATERMARK_STATE_KEY, {"last_ts": last_ts})
    return written


def main() -> None:
    cfg = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    setup_logging(cfg.get("run", {}).get("log_level", "INFO"))

    state_root = Path(cfg["storage"]["state_dir"])
    raw_root = Path(cfg["storage"]["raw_dir"])
    try:
        with SingleInstanceLock(state_root / "locks", "normalize"):
            state = StateStore(state_root)
            written = normalize_ws_raw(raw_root, state)
            logger.info("normalize_app finished written=%s", written)
    except AlreadyRunning:
        logger.warning("normalize already running; skipping this run")


if __name__ == "__main__":
    main()
