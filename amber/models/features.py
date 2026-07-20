from __future__ import annotations

from typing import Any, Sequence

# Canonical model feature set. `mid_price` is deliberately excluded: an absolute
# price level does not generalize across symbols.
MODEL_FEATURES: list[str] = [
    "ret_1",
    "ret_5",
    "ret_20",
    "vol_z_20",
    "oi_z_20",
    "funding_z_20",
    "spread_bps",
]


def feature_vector(row: dict[str, Any], names: Sequence[str] | None = None) -> list[float]:
    return [float(row.get(k, 0.0) or 0.0) for k in (names or MODEL_FEATURES)]
