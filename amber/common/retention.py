"""Disk retention helpers: keep only the most recent run artifacts and drop
raw WS files that have already been fully consumed by the normalizer.
"""

from __future__ import annotations

import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


def dataset_keep(storage_cfg: dict) -> int:
    """How many `dataset_*` runs to retain.

    Datasets dwarf every other artifact (hundreds of MB per hourly rebuild vs a
    few MB for models) and are fully regenerable from `features/` in about a
    minute, so they get their own, much tighter budget. Keeping the default at 5
    like models once let them reach 53 GB on a 59 GB disk.
    """
    raw = storage_cfg.get("keep_dataset_runs")
    if raw is None:
        return 2
    try:
        # Clamp to >=1: unlike other artifacts, "keep 0" here would let the
        # largest directory on disk grow without any bound at all.
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 2


def free_bytes(path: Path) -> int:
    """Free space on the filesystem holding `path` (0 if it cannot be read).

    Walks up to the nearest existing parent so it also works for a directory that
    has not been created yet.
    """
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return int(shutil.disk_usage(p).free)
    except OSError:
        return 0


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def prune_run_dirs(root: Path, prefix: str, keep: int) -> tuple[int, int]:
    """Keep the newest `keep` `<prefix>*` dirs under root, delete the rest.

    Returns (dirs_deleted, bytes_freed). Names are timestamped so a lexical sort
    is chronological; the most recent runs are always retained.
    """
    root = Path(root)
    if keep <= 0 or not root.exists():
        return (0, 0)
    dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    stale = dirs[:-keep] if len(dirs) > keep else []
    freed = 0
    for d in stale:
        freed += _dir_size(d)
        shutil.rmtree(d, ignore_errors=True)
    if stale:
        logger.info("pruned %s %s* dirs under %s (%.1f MB)", len(stale), prefix, root, freed / 1e6)
    return (len(stale), freed)


def cleanup_consumed_ws_raw(raw_root: Path, offsets: dict[str, int]) -> tuple[int, int]:
    """Delete ws_raw part files fully consumed by normalize, except the newest
    per symbol (still being appended). Returns (files_deleted, bytes_freed) and
    mutates `offsets` to drop the deleted keys.
    """
    raw_root = Path(raw_root)
    ws_root = raw_root / "ws_raw"
    if not ws_root.exists():
        return (0, 0)

    deleted = 0
    freed = 0
    for symbol_dir in ws_root.iterdir():
        if not symbol_dir.is_dir():
            continue
        parts = sorted(symbol_dir.glob("part-*.jsonl"))
        if len(parts) <= 1:
            continue  # keep the single/newest file (being written)
        newest = parts[-1]
        for part in parts[:-1]:
            key = f"{symbol_dir.name}/{part.name}"
            try:
                size = part.stat().st_size
            except OSError:
                continue
            if int(offsets.get(key, 0)) >= size and part != newest:
                freed += size
                part.unlink(missing_ok=True)
                offsets.pop(key, None)
                deleted += 1
    if deleted:
        logger.info("removed %s consumed ws_raw files (%.1f MB)", deleted, freed / 1e6)
    return (deleted, freed)
