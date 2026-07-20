from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {lineno}: {exc.msg}") from exc
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
