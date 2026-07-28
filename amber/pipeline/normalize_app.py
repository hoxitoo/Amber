from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from amber.common.config import ConfigLoader
from amber.common.locks import AlreadyRunning, SingleInstanceLock
from amber.common.logging import setup_logging
from amber.common.retention import cleanup_consumed_ws_raw
from amber.exchange.normalizer import BybitNormalizer, gap_fill, step_ms_for_tf
from amber.storage.parquet_sink import ParquetSink
from amber.storage.state_store import StateStore

logger = logging.getLogger(__name__)

OFFSETS_STATE_KEY = "normalize_offsets"
WATERMARK_STATE_KEY = "normalized_watermark"
TRADE_BUCKETS_STATE_KEY = "trade_buckets"
_MINUTE_MS = 60_000
_BUCKET_PRUNE_MS = 30 * _MINUTE_MS  # drop trade buckets older than this vs the newest seen


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

    # Reclaim disk BEFORE writing anything. This used to run at the very end, so
    # a full disk made the write below raise and the cleanup never executed —
    # the one mechanism that frees space was unreachable exactly when it was
    # needed, and the box stayed wedged until a human intervened (audit C1).
    cleanup_consumed_ws_raw(raw_root, offsets)
    state.set(OFFSETS_STATE_KEY, offsets)

    watermark = state.get(WATERMARK_STATE_KEY)
    last_ts: dict[str, int] = {k: int(v) for k, v in watermark.get("last_ts", {}).items()}
    last_row_by_symbol: dict[str, Any] = {}
    # Persist per-minute taker-volume buckets so trades seen in one run still
    # attach to a candle that only closes in a later run.
    trade_buckets: dict[str, dict[str, float]] = {
        k: dict(v) for k, v in state.get(TRADE_BUCKETS_STATE_KEY).items()
    }
    newest_minute = 0

    written = 0
    try:
        for file in sorted((raw_root / "ws_raw").glob("*/part-*.jsonl")):
            symbol = file.parent.name
            key = f"{symbol}/{file.name}"
            payloads, new_offset = _read_new_payloads(file, int(offsets.get(key, 0)))

            # Progress for this file is staged, not applied: a failed write (e.g.
            # a full disk) must leave the offset and watermark untouched so the
            # payloads are re-read next run instead of being silently skipped.
            out: list[dict[str, Any]] = []
            pending_last_ts: dict[str, int] = {}
            consumed_buckets: dict[str, dict[str, float]] = {}
            for payload in payloads:
                if normalizer.update_from_ws_ticker(payload):
                    continue
                trades = normalizer.trades_from_ws(payload)
                if trades:
                    for tsym, tts, side, size in trades:
                        minute = (tts // _MINUTE_MS) * _MINUTE_MS
                        newest_minute = max(newest_minute, minute)
                        bucket = trade_buckets.setdefault(f"{tsym}|{minute}", {"buy": 0.0, "sell": 0.0, "count": 0.0})
                        bucket["buy" if side == "Buy" else "sell"] += size
                        bucket["count"] += 1.0
                    continue
                for candle in normalizer.candles_from_ws(payload):
                    seen_ts = max(int(last_ts.get(symbol, -1)), int(pending_last_ts.get(symbol, -1)))
                    if candle.ts <= seen_ts:
                        continue
                    candle_minute = (candle.ts // _MINUTE_MS) * _MINUTE_MS
                    bucket_key = f"{candle.symbol}|{candle_minute}"
                    agg = trade_buckets.pop(bucket_key, None)
                    if agg is not None:
                        consumed_buckets[bucket_key] = agg
                    row = normalizer.to_normalized(candle, trades=agg)
                    last = last_row_by_symbol.get(symbol)
                    step_ms = step_ms_for_tf(row.tf)
                    if last is not None and row.ts - last.ts > step_ms:
                        out.extend([x.model_dump() for x in gap_fill(last, row.ts, step_ms)])
                    out.append(row.model_dump())
                    last_row_by_symbol[symbol] = row
                    pending_last_ts[symbol] = candle.ts

            try:
                if out:
                    sink.write_records(topic="normalized", symbol=symbol, records=out)
                    written += len(out)
            except OSError:
                # Put back what this file consumed so nothing is lost, then let
                # the caller see the failure (the finally block still persists
                # the progress of the files that did succeed).
                trade_buckets.update(consumed_buckets)
                raise
            offsets[key] = new_offset
            last_ts.update(pending_last_ts)
    finally:
        # Bound the bucket store: drop minutes far behind the newest trade seen.
        if newest_minute:
            cutoff = newest_minute - _BUCKET_PRUNE_MS
            trade_buckets = {k: v for k, v in trade_buckets.items() if int(k.split("|")[1]) >= cutoff}

        state.set(OFFSETS_STATE_KEY, offsets)
        state.set(WATERMARK_STATE_KEY, {"last_ts": last_ts})
        state.set(TRADE_BUCKETS_STATE_KEY, trade_buckets)
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
