from __future__ import annotations

from typing import Sequence

LABEL_SHAPES = ("two_sided", "one_sided")


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


def label_path(
    prices: Sequence[float], up_pct: float, down_pct: float, shape: str = "two_sided"
) -> dict[str, int | None]:
    """Training labels under `shape`, always alongside first-touch fields.

    `shape` decides only what `up_hit`/`down_hit` mean — the target the model
    learns:

    - "two_sided" is the triple barrier: `up_hit` is 1 only when the up barrier
      was reached *first*. This encodes a stop-loss, which is what a mechanical
      trade experiences.
    - "one_sided" asks whether each barrier was reached *at any point*, so a run
      that dips before it rallies still counts as a pump. That is what an alert
      claiming "a pump is coming" actually asserts, and it is the definition
      `quality_report._confirmed_outcome` already scores live signals against.

    Under "one_sided" both flags can be 1 at once, which the triple barrier can
    never produce. Anything deciding a trade outcome must therefore read
    `first_hit`, never infer it from the pair — an `if up_hit ... elif down_hit`
    chain silently books a dip-then-rally as a clean win. `first_hit` and
    `tte_idx` keep first-touch semantics under both shapes precisely so that
    PnL accounting stays correct when the training target changes.
    """
    if shape not in LABEL_SHAPES:
        raise ValueError(f"unknown label shape {shape!r}; expected one of {LABEL_SHAPES}")

    path = label_event_path(prices, up_pct=up_pct, down_pct=down_pct)
    if shape == "two_sided":
        return path

    up = down = 0
    if len(prices) >= 2:
        p0 = prices[0]
        up_level = p0 * (1.0 + up_pct)
        down_level = p0 * (1.0 - down_pct)
        up = int(any(p >= up_level for p in prices[1:]))
        down = int(any(p <= down_level for p in prices[1:]))
    return {"up_hit": up, "down_hit": down, "first_hit": path["first_hit"], "tte_idx": path["tte_idx"]}
