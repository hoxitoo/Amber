from __future__ import annotations

import json
from pathlib import Path


def select_universe(features_root: Path, top_k: int = 20, min_obs: int = 1) -> list[str]:
    """Select symbols by latest volume z-score and observation availability."""
    scores: list[tuple[str, float]] = []
    for f in sorted((features_root / "features").glob("*/part-000.jsonl")):
        symbol = f.parent.name
        lines = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not lines:
            continue
        last = lines[-1]
        if int(last.get("obs", 0)) < min_obs:
            continue
        score = float(last.get("vol_z_20", 0.0))
        scores.append((symbol, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scores[:top_k]]
