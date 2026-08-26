"""Permutation importance and correlation pruning (audit M3).

Answers a question the project has been guessing at: of the 21 features, which
ones actually carry signal? A feature is shuffled within the evaluation segment
and the metric is recomputed — if nothing drops, the model was not using it.
Doing this out-of-sample matters: on training rows a model will "use" noise it
has memorised, so importance measured there would flatter every feature.

Cheap enough to run every retrain, and the answer decides whether adding new
inputs (order book, liquidations) is worth the cost or whether the existing set
needs pruning first.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from amber.models.features import MODEL_FEATURES, feature_vector

logger = logging.getLogger(__name__)

# Below this the drop is indistinguishable from shuffle noise.
USELESS_EPS = 1e-4
CORRELATION_ALERT = 0.95


def _predict_matrix(model: dict[str, Any], head_name: str, matrix: list[list[float]]) -> list[float]:
    """Batch scoring — per-row inference would make permutation testing hours long."""
    heads = model.get("heads", {})
    head = heads.get(head_name, {}) if isinstance(heads, dict) else {}
    kind = head.get("type")

    if kind == "constant":
        return [float(head.get("prob", 0.0))] * len(matrix)
    if kind == "lightgbm":
        import lightgbm as lgb

        booster = head.get("_booster_obj")
        if booster is None:
            booster = lgb.Booster(model_str=head["booster"])
            head["_booster_obj"] = booster
        return [float(p) for p in booster.predict(matrix)]

    import math

    weights = head.get("weights", {})
    bias = float(head.get("bias", 0.0))
    idx = {name: j for j, name in enumerate(model.get("features") or MODEL_FEATURES)}
    out = []
    for row in matrix:
        z = bias + sum(float(w) * row[idx[n]] for n, w in weights.items() if n in idx)
        out.append(1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z)))
    return out


def _pr_auc(y: list[int], p: list[float]) -> float | None:
    if len(set(y)) < 2:
        return None
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y, p))
    except Exception:
        return None


def _clipped_matrix(model: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[list[list[float]], list[str]]:
    names = list(model.get("features") or MODEL_FEATURES)
    bounds = model.get("clip_bounds") if isinstance(model.get("clip_bounds"), dict) else {}
    matrix = []
    for r in rows:
        vec = feature_vector(r, names)
        if bounds:
            vec = [
                min(float(bounds[n][1]), max(float(bounds[n][0]), v)) if n in bounds else v
                for n, v in zip(names, vec)
            ]
        matrix.append(vec)
    return matrix, names


def permutation_importance(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    target: str = "pump",
    label_key: str = "up_hit",
    n_repeats: int = 3,
    seed: int = 7,
) -> dict[str, Any]:
    """Drop in PR-AUC when each feature is shuffled, averaged over `n_repeats`."""
    if not rows:
        return {"status": "no_rows"}
    y = [int(r.get(label_key, 0)) for r in rows]
    if len(set(y)) < 2:
        return {"status": "single_class"}

    matrix, names = _clipped_matrix(model, rows)
    baseline = _pr_auc(y, _predict_matrix(model, target, matrix))
    if baseline is None:
        return {"status": "metric_unavailable"}

    rng = random.Random(seed)
    scores: list[dict[str, Any]] = []
    for j, name in enumerate(names):
        column = [row[j] for row in matrix]
        drops = []
        for _ in range(max(1, n_repeats)):
            shuffled = column[:]
            rng.shuffle(shuffled)
            for row, v in zip(matrix, shuffled):
                row[j] = v
            permuted = _pr_auc(y, _predict_matrix(model, target, matrix))
            drops.append(baseline - permuted if permuted is not None else 0.0)
        for row, v in zip(matrix, column):  # restore before moving on
            row[j] = v
        mean_drop = sum(drops) / len(drops)
        scores.append({
            "feature": name,
            "importance": mean_drop,
            "importance_pct": (mean_drop / baseline * 100.0) if baseline else 0.0,
            "useless": mean_drop <= USELESS_EPS,
        })

    scores.sort(key=lambda s: s["importance"], reverse=True)
    useless = [s["feature"] for s in scores if s["useless"]]
    carrying = [s for s in scores if not s["useless"]]
    top_share = (
        sum(s["importance"] for s in scores[:5]) / sum(s["importance"] for s in carrying) * 100.0
        if carrying and sum(s["importance"] for s in carrying) > 0
        else 0.0
    )
    return {
        "status": "ok",
        "target": target,
        "rows": len(rows),
        "baseline_pr_auc": baseline,
        "n_repeats": n_repeats,
        "scores": scores,
        "useless_features": useless,
        "carrying_count": len(carrying),
        "top5_share_pct": top_share,
    }


def correlated_pairs(rows: list[dict[str, Any]], threshold: float = CORRELATION_ALERT) -> list[dict[str, Any]]:
    """Feature pairs so collinear that one of them is redundant."""
    if len(rows) < 30:
        return []
    cols = {n: [float(r.get(n, 0.0) or 0.0) for r in rows] for n in MODEL_FEATURES}

    def corr(a: list[float], b: list[float]) -> float:
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        va = sum((x - ma) ** 2 for x in a)
        vb = sum((x - mb) ** 2 for x in b)
        if va <= 0 or vb <= 0:
            return 0.0
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        return cov / (va**0.5 * vb**0.5)

    names = list(MODEL_FEATURES)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            r = corr(cols[a], cols[b])
            if abs(r) >= threshold:
                out.append({"a": a, "b": b, "corr": r})
    return sorted(out, key=lambda d: abs(d["corr"]), reverse=True)
