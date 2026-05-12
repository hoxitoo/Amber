from __future__ import annotations

from typing import Sequence


def label_event_path(prices: Sequence[float], up_pct: float, down_pct: float) -> dict[str, int | None]:
    """Label if +X% or -X% barrier is hit first within horizon.

    Returns
    - up_hit: 1/0
    - down_hit: 1/0
    - first_hit: 1 (up), -1 (down), 0 (none)
    - tte_idx: index of first hit relative to start, else None
    """
    if len(prices) < 2:
        return {"up_hit": 0, "down_hit": 0, "first_hit": 0, "tte_idx": None}

    p0 = prices[0]
    up_level = p0 * (1.0 + up_pct)
    down_level = p0 * (1.0 - down_pct)

    for i, p in enumerate(prices[1:], start=1):
        if p >= up_level:
            return {"up_hit": 1, "down_hit": 0, "first_hit": 1, "tte_idx": i}
        if p <= down_level:
            return {"up_hit": 0, "down_hit": 1, "first_hit": -1, "tte_idx": i}

    return {"up_hit": 0, "down_hit": 0, "first_hit": 0, "tte_idx": None}
