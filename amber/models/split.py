from __future__ import annotations

from dataclasses import dataclass


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
