from __future__ import annotations

import json
from pathlib import Path


def select_universe(
    features_root: Path,
    top_k: int = 20,
    min_obs: int = 1,
    min_dollar_volume: float = 0.0,
) -> list[str]:
    """Select symbols by latest volume z-score with minimum data/liquidity guards."""
    scores: list[tuple[str, float]] = []
    by_symbol: dict[str, Path] = {}
    for f in sorted((features_root / "features").glob("*/part-*.jsonl")):
        by_symbol[f.parent.name] = f
    for symbol, f in sorted(by_symbol.items()):
        lines = []
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not lines:
            continue
        last = lines[-1]
        if int(last.get("obs", 0)) < min_obs:
            continue
        if float(last.get("notional_volume_1m", 0.0)) < min_dollar_volume:
            continue
        score = float(last.get("vol_z_20", 0.0))
        scores.append((symbol, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scores[:top_k]]
