from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from amber.features.online import FeatureEngine
from amber.storage.state_store import StateStore

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skip invalid jsonl line file=%s line=%s", path, lineno)
    return rows


def _read_normalized_rows(raw_root: Path, symbol: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for part in sorted((raw_root / "normalized" / symbol).glob("part-*.jsonl")):
        rows.extend(_read_jsonl(part))
    return rows


def compute_batch_features(
    raw_root: Path,
    out_root: Path,
    symbols: list[str],
    state: StateStore | None = None,  # kept for call compatibility; no longer used
) -> dict[str, int]:
    """Recompute the full feature file for every symbol from sorted normalized rows.

    The file is rewritten atomically each run rather than appended to. Normalized
    history can grow non-append (REST backfill fills candles *behind* the live WS
    watermark), which an append-only + count-watermark scheme would either
    duplicate or miss. A full recompute is always correct and, at local scale
    (thousands of rows/symbol), cheap. Partitioned storage for large scale is the
    Sprint-3 backlog item (A1).
    """
    out_root.mkdir(parents=True, exist_ok=True)

    written = 0
    for symbol in symbols:
        rows = _read_normalized_rows(raw_root, symbol)
        if not rows:
            continue
        rows.sort(key=lambda r: int(r.get("ts", 0) or 0))

        engine = FeatureEngine()
        feature_rows = engine.transform_rows(rows)

        target = out_root / "features" / symbol
        target.mkdir(parents=True, exist_ok=True)
        tmp = target / "part-000.jsonl.tmp"
        with tmp.open("w", encoding="utf-8") as fh:
            for feature_row in feature_rows:
                fh.write(json.dumps(feature_row, ensure_ascii=False) + "\n")
        tmp.replace(target / "part-000.jsonl")  # atomic swap
        written += len(feature_rows)

    return {"written_rows": written}
