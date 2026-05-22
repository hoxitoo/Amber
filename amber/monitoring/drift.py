from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - fallback if sklearn unavailable
    roc_auc_score = None


def detect_drift(features_root: Path, symbol: str, threshold: float = 0.05) -> dict[str, float | bool]:
    path = features_root / "features" / symbol / "part-000.jsonl"
    if not path.exists():
        return {"drift": False, "delta_ret_mean": 0.0}

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if len(rows) < 4:
        return {"drift": False, "delta_ret_mean": 0.0}

    mid = len(rows) // 2
    m1 = sum(float(r.get("ret_1", 0.0)) for r in rows[:mid]) / max(1, mid)
    m2 = sum(float(r.get("ret_1", 0.0)) for r in rows[mid:]) / max(1, len(rows) - mid)
    delta = abs(m2 - m1)
    return {"drift": delta > threshold, "delta_ret_mean": delta}


class RollingAUCMonitor:
    def __init__(self, window: int = 200) -> None:
        self.window = window
        self.y_true: deque[int] = deque(maxlen=window)
        self.y_score: deque[float] = deque(maxlen=window)

    def update(self, y_true: int, y_score: float) -> None:
        self.y_true.append(int(y_true))
        self.y_score.append(float(y_score))

    def value(self) -> float | None:
        if len(self.y_true) < 20:
            return None
        if roc_auc_score is None:
            return None
        if len(set(self.y_true)) < 2:
            return None
        return float(roc_auc_score(list(self.y_true), list(self.y_score)))


class PredictionBiasMonitor:
    def __init__(self, window: int = 200) -> None:
        self.window = window
        self.up_probs: deque[float] = deque(maxlen=window)
        self.down_probs: deque[float] = deque(maxlen=window)

    def update(self, p_up: float, p_down: float) -> None:
        self.up_probs.append(float(p_up))
        self.down_probs.append(float(p_down))

    def bias(self) -> float | None:
        if len(self.up_probs) < 20:
            return None
        return mean(self.up_probs) - mean(self.down_probs)


def psi(expected: list[float], actual: list[float], bins: int = 10) -> float:
    if not expected or not actual:
        return 0.0
    lo = min(min(expected), min(actual))
    hi = max(max(expected), max(actual))
    if hi == lo:
        return 0.0

    step = (hi - lo) / bins

    def hist(vals: list[float]) -> list[float]:
        counts = [0] * bins
        for v in vals:
            i = min(bins - 1, int((v - lo) / step))
            counts[i] += 1
        total = max(1, len(vals))
        return [max(1e-6, c / total) for c in counts]

    e = hist(expected)
    a = hist(actual)
    return sum((av - ev) * math.log(av / ev) for ev, av in zip(e, a))


def psi_monitor(reference: list[float], live: list[float]) -> dict[str, float | str]:
    v = psi(reference, live)
    if v > 0.2:
        level = "high"
    elif v > 0.1:
        level = "medium"
    else:
        level = "low"
    return {"psi": float(v), "level": level}
