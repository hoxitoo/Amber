from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Fold:
    train_start: int
    train_end: int
    val_start: int
    val_end: int


def make_walk_forward_folds(n_rows: int, n_folds: int = 3, val_size: int = 50, gap: int = 30) -> list[Fold]:
    """Create chronological walk-forward folds with explicit gap.

    Indices are [start, end) boundaries.
    """
    if n_rows <= (val_size + gap + 10):
        return []

    folds: list[Fold] = []
    step = max(1, (n_rows - (val_size + gap)) // n_folds)

    for i in range(1, n_folds + 1):
        train_end = step * i
        val_start = train_end + gap
        val_end = min(val_start + val_size, n_rows)
        if val_end - val_start < 5:
            continue
        folds.append(Fold(train_start=0, train_end=train_end, val_start=val_start, val_end=val_end))

    return folds


def make_time_walk_forward_folds(
    pseudo_ts: list[int],
    n_folds: int = 3,
    val_size: int = 50,
    gap: int = 30,
) -> list[dict[str, int]]:
    """Walk-forward folds expressed as pseudo-time boundaries.

    `val_size` and `gap` are counted in distinct pseudo-time steps (candles),
    not rows — with multi-horizon/multi-symbol datasets several rows share one
    timestamp, and a row-count gap would not span the intended time distance.

    Each fold dict has inclusive boundaries:
    train: t <= train_end; val: val_start <= t <= val_end.
    """
    uniq = sorted(set(pseudo_ts))
    idx_folds = make_walk_forward_folds(len(uniq), n_folds=n_folds, val_size=val_size, gap=gap)
    out: list[dict[str, int]] = []
    for f in idx_folds:
        out.append(
            {
                "train_end": uniq[f.train_end - 1],
                "val_start": uniq[f.val_start],
                "val_end": uniq[f.val_end - 1],
            }
        )
    return out


def make_holdout_splits(
    pseudo_ts: list[int],
    *,
    train_frac: float = 0.7,
    calib_frac: float = 0.15,
    gap: int = 30,
) -> dict[str, Any] | None:
    """Chronological train / calibration-holdout / test boundaries with purge gaps.

    Boundaries are inclusive pseudo-time values; `gap` distinct time steps are
    dropped between segments so labels whose forward window crosses a boundary
    cannot leak. Returns None when there is not enough data for an honest split
    (callers then fall back to in-sample mode and must flag it).
    """
    if not (0.0 < train_frac < 1.0) or not (0.0 < calib_frac < 1.0) or train_frac + calib_frac >= 1.0:
        raise ValueError("train_frac/calib_frac must be in (0,1) and sum below 1")
    uniq = sorted(set(pseudo_ts))
    n = len(uniq)
    train_end_i = int(n * train_frac) - 1
    calib_start_i = train_end_i + 1 + gap
    calib_end_i = calib_start_i + max(1, int(n * calib_frac)) - 1
    test_start_i = calib_end_i + 1 + gap
    if train_end_i < 10 or test_start_i > n - 5:
        return None
    return {
        "train_end": uniq[train_end_i],
        "calib_start": uniq[calib_start_i],
        "calib_end": uniq[calib_end_i],
        "test_start": uniq[test_start_i],
        "gap_candles": gap,
    }
