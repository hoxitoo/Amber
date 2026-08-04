from __future__ import annotations

from bisect import bisect_right
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


def _read_feature_rows(features_root: Path, symbol: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((features_root / "features" / symbol).glob("part-*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def detect_drift(
    features_root: Path,
    symbol: str,
    threshold: float = 0.2,
    *,
    reference: dict[str, list[float]] | None = None,
    window: int = 500,
) -> dict[str, Any]:
    """Feature drift via PSI.

    With a train `reference` (quantile edges per feature, stored in the model
    artifact as `train_reference`), recent live values are compared against the
    training distribution. Without one, it degrades to split-half PSI on `ret_1`.
    Returns drift=True when max PSI exceeds `threshold` (0.1 warn / 0.2 alert
    are the conventional PSI levels).
    """
    rows = _read_feature_rows(features_root, symbol)
    ret_vals = [float(r.get("ret_1", 0.0)) for r in rows]
    mid = len(ret_vals) // 2
    delta_ret_mean = 0.0
    if len(ret_vals) >= 4:
        m1 = sum(ret_vals[:mid]) / max(1, mid)
        m2 = sum(ret_vals[mid:]) / max(1, len(ret_vals) - mid)
        delta_ret_mean = abs(m2 - m1)

    per_feature: dict[str, float] = {}
    if reference:
        recent = rows[-window:]
        for name, edges in reference.items():
            live = [float(r.get(name, 0.0) or 0.0) for r in recent]
            if len(live) >= 20:
                per_feature[name] = psi_from_quantile_reference(edges, live)
    elif len(ret_vals) >= 4:
        per_feature["ret_1"] = psi(ret_vals[:mid], ret_vals[mid:])

    max_psi = max(per_feature.values()) if per_feature else 0.0
    level = "high" if max_psi > 0.2 else "medium" if max_psi > 0.1 else "low"
    return {
        "drift": max_psi > threshold,
        "max_psi": max_psi,
        "level": level,
        "per_feature": per_feature,
        "reference": "train_quantiles" if reference else "split_half",
        "delta_ret_mean": delta_ret_mean,
    }


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


def psi_from_quantile_reference(
    reference: list[float] | dict[str, list[float]], live: list[float], eps: float = 1e-6
) -> float:
    """PSI of `live` against a train-set reference.

    `reference` is either {"edges": [...], "expected": [...]} — quantile
    boundaries plus the train share actually observed in each bin — or, for
    legacy artifacts, a bare list of edges.

    The expected share is only uniform (1/n_bins) when the quantile edges are
    distinct. Discrete features break that: `breakout_up_20` is ~95% zeros, so
    its deciles are [0,0,...,0,1] and every live value falls into one bin,
    producing PSI ≈ 12.434 — the exact figure the dashboard was reporting — even
    for a distribution identical to train. Storing the real per-bin shares fixes
    it; legacy references with tied edges cannot be corrected after the fact, so
    they report no drift rather than a fabricated alarm (audit B4b).
    """
    if isinstance(reference, dict) and reference.get("values"):
        # Categorical reference: quantile edges cannot separate 0 from 1, so
        # low-cardinality features are scored by exact value, with anything
        # unseen in training pooled into an extra bucket so it registers.
        values = [float(x) for x in reference["values"]]
        expected = [float(x) for x in reference.get("expected", [])]
        if len(expected) != len(values) or not live:
            return 0.0
        index = {v: i for i, v in enumerate(values)}
        counts = [0] * (len(values) + 1)
        for v in live:
            counts[index.get(float(v), len(values))] += 1
        total = len(live)
        out = 0.0
        for c, e in zip(counts, [*expected, 0.0]):
            a = max(eps, c / total)
            out += (a - max(eps, e)) * math.log(a / max(eps, e))
        return out

    if isinstance(reference, dict):
        edges = [float(x) for x in reference.get("edges", [])]
        expected = [float(x) for x in reference.get("expected", [])]
    else:
        edges = [float(x) for x in reference]
        expected = []

    n_bins = len(edges) - 1
    if n_bins < 1 or not live:
        return 0.0
    if edges[-1] <= edges[0]:
        return 0.0  # constant train feature: no reference distribution exists
    if len(expected) != n_bins:
        if len(set(edges)) != len(edges):
            return 0.0  # tied edges without stored shares — cannot be scored honestly
        expected = [1.0 / n_bins] * n_bins

    counts = [0] * n_bins
    for v in live:
        idx = bisect_right(edges, v) - 1
        counts[max(0, min(n_bins - 1, idx))] += 1

    total = len(live)
    out = 0.0
    for c, exp in zip(counts, expected):
        a = max(eps, c / total)
        e = max(eps, exp)
        out += (a - e) * math.log(a / e)
    return out


def psi_monitor(reference: list[float], live: list[float]) -> dict[str, float | str]:
    v = psi(reference, live)
    if v > 0.2:
        level = "high"
    elif v > 0.1:
        level = "medium"
    else:
        level = "low"
    return {"psi": float(v), "level": level}
