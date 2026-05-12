from __future__ import annotations

import json
from pathlib import Path


def detect_drift(features_root: Path, symbol: str, threshold: float = 0.05) -> dict[str, float | bool]:
    path = features_root / "features" / symbol / "part-000.jsonl"
    if not path.exists():
        return {"drift": False, "delta_ret_mean": 0.0}

    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) < 4:
        return {"drift": False, "delta_ret_mean": 0.0}

    mid = len(rows) // 2
    m1 = sum(float(r.get("ret_1m", 0.0)) for r in rows[:mid]) / max(1, mid)
    m2 = sum(float(r.get("ret_1m", 0.0)) for r in rows[mid:]) / max(1, len(rows) - mid)
    delta = abs(m2 - m1)
    return {"drift": delta > threshold, "delta_ret_mean": delta}
