from __future__ import annotations

import json
from pathlib import Path


<<<<<<< HEAD
def select_universe(
    features_root: Path,
    top_k: int = 20,
    min_obs: int = 1,
    min_dollar_volume: float = 0.0,
) -> list[str]:
    """Select symbols by latest volume z-score with minimum data/liquidity guards."""
    scores: list[tuple[str, float]] = []
    for f in sorted((features_root / "features").glob("*/part-000.jsonl")):
        symbol = f.parent.name
        lines = []
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue
=======
def select_universe(features_root: Path, top_k: int = 20, min_obs: int = 1) -> list[str]:
    """Select symbols by latest volume z-score and observation availability."""
    scores: list[tuple[str, float]] = []
    for f in sorted((features_root / "features").glob("*/part-000.jsonl")):
        symbol = f.parent.name
        lines = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
>>>>>>> origin/main
        if not lines:
            continue
        last = lines[-1]
        if int(last.get("obs", 0)) < min_obs:
            continue
<<<<<<< HEAD
        if float(last.get("notional_volume_1m", 0.0)) < min_dollar_volume:
            continue
=======
>>>>>>> origin/main
        score = float(last.get("vol_z_20", 0.0))
        scores.append((symbol, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scores[:top_k]]
