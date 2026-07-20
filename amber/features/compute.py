from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from amber.features.online import FeatureEngine
from amber.storage.state_store import StateStore

logger = logging.getLogger(__name__)

FEATURES_WATERMARK_KEY = "features_watermark"


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
    state: StateStore | None = None,
) -> dict[str, int]:
    """Recompute features for every normalized row and append only new rows.

    The full normalized history is replayed through the FeatureEngine (rolling
    windows need it), but rows already written in previous runs are skipped via a
    per-symbol watermark, keeping the stage idempotent.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    watermark: dict[str, int] = {}
    if state is not None:
        watermark = {k: int(v) for k, v in state.get(FEATURES_WATERMARK_KEY).items()}

    written = 0
    for symbol in symbols:
        rows = _read_normalized_rows(raw_root, symbol)
        if not rows:
            continue
        rows.sort(key=lambda r: int(r.get("ts", 0)))

        engine = FeatureEngine()
        feature_rows = engine.transform_rows(rows)
        already = min(int(watermark.get(symbol, 0)), len(feature_rows))
        new_rows = feature_rows[already:]
        if not new_rows:
            continue

        target = out_root / "features" / symbol
        target.mkdir(parents=True, exist_ok=True)
        with (target / "part-000.jsonl").open("a", encoding="utf-8") as fh:
            for feature_row in new_rows:
                fh.write(json.dumps(feature_row, ensure_ascii=False) + "\n")
        written += len(new_rows)
        watermark[symbol] = len(feature_rows)

    if state is not None:
        state.set(FEATURES_WATERMARK_KEY, watermark)
    return {"written_rows": written}
