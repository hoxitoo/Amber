from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL artifact, rejecting corruption.

    One exception: a *truncated* final line — the last line of the file, with no
    terminating newline — is dropped with a warning. That is the signature of a
    writer killed mid-line (a full disk, a hard kill), and discarding a 200k-row
    dataset over its incomplete tail record is worse than losing that record. A
    malformed line that IS newline-terminated was written in full, so it means
    real corruption and still raises, as does corruption anywhere earlier.

    The file is streamed, never slurped: datasets run to hundreds of MB and this
    is called several times per retrain (train, calibrate, eval, backtest), so
    holding the raw text in memory on top of the parsed rows is what pushes a
    small VPS into the OOM killer. A malformed line is therefore held as pending
    rather than judged immediately — only reaching EOF proves it was last.
    """
    rows: list[dict[str, Any]] = []
    pending: tuple[int, str, str] | None = None
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if pending is not None:
                # More data followed the malformed line, so it was not a
                # truncated tail: this is real corruption.
                bad_lineno, _, msg = pending
                raise ValueError(f"Invalid JSONL in {path} at line {bad_lineno}: {msg}")
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                pending = (lineno, line, exc.msg)
    if pending is not None:
        bad_lineno, bad_line, msg = pending
        if bad_line.endswith("\n"):
            # Written in full, so the damage is real rather than a clean cut.
            raise ValueError(f"Invalid JSONL in {path} at line {bad_lineno}: {msg}")
        logger.warning("dropping truncated final line in %s: %s", path, msg)
    return rows


def latest_dataset_dir(datasets_root: Path) -> Path:
    if not datasets_root.exists():
        raise ValueError(f"No dataset_* directories found under: {datasets_root}")
    candidates = sorted([p for p in datasets_root.iterdir() if p.is_dir() and p.name.startswith("dataset_")])
    if not candidates:
        raise ValueError(f"No dataset_* directories found under: {datasets_root}")
    return candidates[-1]


def load_latest_dataset_rows(datasets_root: Path) -> tuple[list[dict[str, Any]], str]:
    latest = latest_dataset_dir(datasets_root)
    dataset_file = latest / "dataset.jsonl"
    if not dataset_file.exists():
        raise ValueError(f"Missing dataset file: {dataset_file}")
    return read_jsonl_strict(dataset_file), latest.name


def order_with_pseudo_time(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int], str]:
    """Return rows in chronological order plus a pseudo-time axis.

    When the dataset carries real timestamps, rows are sorted by
    (ts, symbol, horizon) and pseudo-time is the timestamp — this is what makes
    walk-forward splits leakage-safe across interleaved symbols/horizons. Tiny
    or ts-less datasets fall back to file order with the row index as
    pseudo-time (mode "index").
    """
    ts = [int(r.get("ts") or 0) for r in rows]
    if len(set(ts)) >= 20:
        order = sorted(
            range(len(rows)),
            key=lambda i: (ts[i], str(rows[i].get("symbol", "")), int(rows[i].get("horizon_steps", 0) or 0)),
        )
        ordered = [rows[i] for i in order]
        return ordered, [int(r.get("ts") or 0) for r in ordered], "ts"
    return rows, list(range(len(rows))), "index"


def split_rows(
    rows: list[dict[str, Any]],
    pseudo_ts: list[int],
    splits: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Partition ordered rows into train/calib/test segments by pseudo-time."""
    train_end = int(splits["train_end"])
    calib_start = int(splits["calib_start"])
    calib_end = int(splits["calib_end"])
    test_start = int(splits["test_start"])

    out: dict[str, list[dict[str, Any]]] = {"train": [], "calib": [], "test": []}
    for row, t in zip(rows, pseudo_ts):
        if t <= train_end:
            out["train"].append(row)
        elif calib_start <= t <= calib_end:
            out["calib"].append(row)
        elif t >= test_start:
            out["test"].append(row)
    return out
