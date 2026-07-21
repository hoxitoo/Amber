from __future__ import annotations

from typing import Any, Sequence

# Canonical model feature set. Absolute price levels (mid_price/bid/ask) are
# deliberately excluded — they do not generalize across symbols. Everything here
# is scale-free so a single model works across the whole universe.
MODEL_FEATURES: list[str] = [
    # returns (short -> long context)
    "ret_1",
    "ret_5",
    "ret_20",
    "ret_60",
    # volume surge & acceleration (pumps start here)
    "vol_z_20",
    "vol_ratio_20",
    "vol_accel",
    # open interest positioning & funding
    "oi_z_20",
    "oi_roc_5",
    "funding_z_20",
    # volatility compression -> expansion (squeeze before breakout)
    "squeeze_ratio",
    "bb_width_20",
    "range_atr_14",
    # breakout geometry (resistance/support break the user draws by hand)
    "dist_to_high_20",
    "dist_to_low_20",
    "breakout_up_20",
    "breakout_dn_20",
    # microstructure
    "spread_bps",
]


def feature_vector(row: dict[str, Any], names: Sequence[str] | None = None) -> list[float]:
    return [float(row.get(k, 0.0) or 0.0) for k in (names or MODEL_FEATURES)]
