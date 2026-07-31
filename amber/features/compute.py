from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
from typing import Any

from amber.features.online import _MAXLEN, FeatureEngine
from amber.features.spec import FEATURE_SPEC_VERSION
from amber.storage.state_store import StateStore

logger = logging.getLogger(__name__)

# Rows to replay before the resume point so the engine's rolling windows are in
# exactly the state a full recompute would leave them in. The engine keeps all
# state in deques bounded by _MAXLEN and every feature reads only those deques,
# so replaying the last _MAXLEN rows reproduces the full-history state exactly.
_WARMUP_ROWS = _MAXLEN


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


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            n += chunk.count(b"\n")
    return n


def _resume_index(out_file: Path, meta_file: Path, rows: list[dict[str, Any]]) -> int:
    """How many leading feature rows can be kept as-is (0 = full recompute).

    Only an exact prefix match is trusted: the stored row count must still line
    up with the same timestamp in the normalized history, and the file must
    really hold that many lines. Anything else — REST backfill inserting candles
    *behind* the live stream, a feature-spec bump, a half-written file — falls
    back to recomputing everything, which is always correct.
    """
    if not out_file.exists() or not meta_file.exists():
        return 0
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    if meta.get("spec_version") != FEATURE_SPEC_VERSION:
        return 0
    try:
        kept = int(meta.get("rows", 0))
        last_ts = int(meta.get("last_ts", -1))
    except (TypeError, ValueError):
        return 0
    if kept <= 0 or kept > len(rows):
        return 0
    if int(rows[kept - 1].get("ts", -2) or -2) != last_ts:
        return 0
    if _count_lines(out_file) != kept:
        return 0
    return kept


def _write_all(out_file: Path, feature_rows: list[dict[str, Any]]) -> None:
    tmp = out_file.with_suffix(".jsonl.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            for row in feature_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(out_file)
    finally:
        tmp.unlink(missing_ok=True)


def _append_new(out_file: Path, new_rows: list[dict[str, Any]]) -> None:
    """Append to the kept prefix without rewriting it, still atomically."""
    if not new_rows:
        return
    tmp = out_file.with_suffix(".jsonl.tmp")
    try:
        shutil.copyfile(out_file, tmp)
        with tmp.open("a", encoding="utf-8") as fh:
            for row in new_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(out_file)
    finally:
        tmp.unlink(missing_ok=True)


def compute_batch_features(
    raw_root: Path,
    out_root: Path,
    symbols: list[str],
    state: StateStore | None = None,  # kept for call compatibility; no longer used
) -> dict[str, int]:
    """Compute features for every symbol, reusing work already on disk.

    Features are append-only in the common case: a candle's feature row never
    changes once the candles around it are known, so only rows past the last
    computed one need work. Recomputing the whole history every cycle made the
    cost grow with accumulated data — at 27 symbols a cycle went from ~4s to
    ~38s over a few days and would eventually exceed the loop interval, pinning
    the CPU. Verified equivalent to a full recompute; falls back to one whenever
    the stored prefix cannot be trusted.
    """
    out_root.mkdir(parents=True, exist_ok=True)

    written = 0
    for symbol in symbols:
        rows = _read_normalized_rows(raw_root, symbol)
        if not rows:
            continue
        rows.sort(key=lambda r: int(r.get("ts", 0) or 0))

        target = out_root / "features" / symbol
        target.mkdir(parents=True, exist_ok=True)
        out_file = target / "part-000.jsonl"
        meta_file = target / "meta.json"

        start = _resume_index(out_file, meta_file, rows)
        engine = FeatureEngine()
        if start > 0:
            for row in rows[max(0, start - _WARMUP_ROWS) : start]:
                engine.update(row)  # warm the windows, emit nothing
            _append_new(out_file, [engine.update(row) for row in rows[start:]])
        else:
            _write_all(out_file, engine.transform_rows(rows))

        meta_file.write_text(
            json.dumps({
                "spec_version": FEATURE_SPEC_VERSION,
                "rows": len(rows),
                "last_ts": int(rows[-1].get("ts", 0) or 0),
            }),
            encoding="utf-8",
        )
        written += len(rows)

    return {"written_rows": written}
