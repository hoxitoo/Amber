from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path

from amber.common.audit_log import log_universe
from amber.common.config import ConfigLoader
from amber.common.locks import AlreadyRunning, SingleInstanceLock
from amber.common.logging import setup_logging
from amber.common.manifest import ArtifactManifest, new_run_id, write_manifest
from amber.exchange.bybit_public import MAINNET_REST_URL, TESTNET_REST_URL, BybitPublicClient
from amber.exchange.normalizer import BybitNormalizer, gap_fill, step_ms_for_tf
from amber.pipeline.normalize_app import WATERMARK_STATE_KEY
from amber.storage.parquet_sink import ParquetSink
from amber.storage.state_store import StateStore

logger = logging.getLogger(__name__)


def _existing_normalized_ts(raw_root: Path, symbol: str) -> set[int]:
    """Timestamps already present in the normalized series for a symbol."""
    seen: set[int] = set()
    for part in sorted((raw_root / "normalized" / symbol).glob("part-*.jsonl")):
        with part.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    seen.add(int(json.loads(line)["ts"]))
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    continue
    return seen


def collect_rest_backfill(
    client: BybitPublicClient,
    sink: ParquetSink,
    state: StateStore,
    symbols: list[str],
    *,
    interval: str = "1",
    limit: int = 200,
    raw_root: Path | None = None,
) -> int:
    """Backfill historical klines per symbol via REST into `normalized/`.

    Dedup is by the set of timestamps already stored, NOT by the forward
    `normalized_watermark`: REST returns *past* candles, and the WS normalize
    stage advances that watermark to the present, so a forward-only skip would
    drop the entire backfill. Deduping on actual stored ts lets backfill fill the
    history *behind* the live stream without duplicating anything.
    """
    normalizer = BybitNormalizer()
    watermark = state.get(WATERMARK_STATE_KEY)
    last_ts: dict[str, int] = {k: int(v) for k, v in watermark.get("last_ts", {}).items()}

    emitted = 0
    for symbol in symbols:
        try:
            ticker, oi, funding = client.get_market_snapshot(symbol)
            normalizer.update_ticker(ticker)
            normalizer.update_oi(oi)
            normalizer.update_funding(funding)
            candles = client.get_klines(symbol, interval=interval, limit=limit)
        except Exception as exc:
            logger.error("collector failed for symbol=%s err=%s", symbol, exc)
            continue

        # The most recent kline from REST is the still-open candle; drop it so only
        # closed candles enter the normalized series (matches WS confirm-only).
        if candles:
            candles = candles[:-1]

        seen = _existing_normalized_ts(raw_root, symbol) if raw_root is not None else set()
        rows = []
        prev = None
        for candle in candles:
            if candle.ts in seen:
                prev = None  # don't gap-fill across an already-present candle
                continue
            row = normalizer.to_normalized(candle)
            step_ms = step_ms_for_tf(row.tf)
            if prev is not None and row.ts - prev.ts > step_ms:
                rows.extend([x.model_dump() for x in gap_fill(prev, row.ts, step_ms)])
            rows.append(row.model_dump())
            prev = row
            last_ts[symbol] = max(last_ts.get(symbol, -1), candle.ts)

        if rows:
            sink.write_records(topic="normalized", symbol=symbol, records=rows)
            emitted += len(rows)

    # Never lower the watermark: the WS stream owns forward progress.
    state.set(WATERMARK_STATE_KEY, {"last_ts": last_ts})
    return emitted


def main() -> None:
    config = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    setup_logging(config.get("run", {}).get("log_level", "INFO"))

    run_id = new_run_id(prefix="collector")
    data_root = Path(config["storage"]["raw_dir"])
    state_root = Path(config["storage"]["state_dir"])
    logs_root = Path(config["storage"]["logs_dir"])
    bybit_cfg = config["exchange"]["bybit"]
    symbols = bybit_cfg["symbols"]
    rest_url = bybit_cfg.get("rest_url") or (TESTNET_REST_URL if bybit_cfg.get("testnet") else MAINNET_REST_URL)
    backfill_limit = int(bybit_cfg.get("backfill_limit", 200))

    try:
        with SingleInstanceLock(state_root / "locks", "collector"):
            sink = ParquetSink(data_root)
            state = StateStore(state_root)
            # Record the in-scope universe for this run (Q1: auditable selection).
            log_universe(logs_root, "collector", symbols, extra={"run_id": run_id})

            with BybitPublicClient(rest_url=rest_url) as client:
                emitted = collect_rest_backfill(
                    client, sink, state, symbols, limit=backfill_limit, raw_root=data_root
                )

            state.set("collector", {"run_id": run_id, "emitted_records": emitted})
            manifest = ArtifactManifest(
                run_id=run_id,
                artifact_type="collector_run",
                artifact_version="v1",
                created_at=datetime.now(timezone.utc).isoformat(),
                config_ref="config/amber.yaml",
                feature_spec_ref="config/features.yaml",
                metadata={"symbols": symbols, "emitted_records": emitted, "topic": "normalized"},
            )
            write_manifest(data_root / "manifests" / run_id / "manifest.json", manifest)
            logger.info("collector finished run_id=%s emitted=%s", run_id, emitted)
    except AlreadyRunning:
        logger.warning("collector already running; skipping this run")


if __name__ == "__main__":
    main()
