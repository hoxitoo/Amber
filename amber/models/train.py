from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from amber.common.manifest import ArtifactManifest, new_run_id, write_manifest
from amber.models.dataset_io import load_latest_dataset_rows, order_with_pseudo_time, split_rows
from amber.models.features import MODEL_FEATURES, feature_vector
from amber.models.split import make_holdout_splits, make_time_walk_forward_folds

logger = logging.getLogger(__name__)

_CLASS_EPS = 1e-4


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _clip_bounds(x: list[list[float]], lo_pct: float = 0.01, hi_pct: float = 0.99) -> dict[str, list[float]]:
    """Per-feature winsorization bounds (1st/99th pct) computed on train data.

    Unbounded ratio features (vol_ratio, oi_roc, ...) spike; clipping stabilizes
    the logreg fallback and keeps PSI reference quantiles meaningful (audit M4).
    """
    out: dict[str, list[float]] = {}
    n = len(x)
    if n < 20:
        return out
    for j, name in enumerate(MODEL_FEATURES):
        col = sorted(row[j] for row in x)
        lo = col[int((n - 1) * lo_pct)]
        hi = col[int((n - 1) * hi_pct)]
        if hi > lo:
            out[name] = [float(lo), float(hi)]
    return out


def _apply_clip(x: list[list[float]], bounds: dict[str, list[float]]) -> list[list[float]]:
    if not bounds:
        return x
    idx = {name: j for j, name in enumerate(MODEL_FEATURES)}
    clipped = [row[:] for row in x]
    for name, (lo, hi) in bounds.items():
        j = idx.get(name)
        if j is None:
            continue
        for row in clipped:
            row[j] = min(hi, max(lo, row[j]))
    return clipped


def _fit_head(x: list[list[float]], y: list[int], w: list[float] | None = None) -> dict[str, Any]:
    """Fit one event head. LightGBM when available, logistic regression as
    fallback, constant prior when the labels are single-class.

    `w` are sample weights (1/horizon uniqueness proxy for overlapping event
    windows, audit Q3). Class imbalance is handled via scale_pos_weight (audit
    M2); the resulting raw scores are rank-oriented and are calibrated
    downstream before any decision is made.
    """
    pos = sum(y)
    if pos == 0 or pos == len(y):
        rate = pos / len(y) if y else 0.0
        return {"type": "constant", "prob": min(1.0 - _CLASS_EPS, max(_CLASS_EPS, rate)), "label_rate": rate}

    try:
        import lightgbm as lgb
        import numpy as np

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "num_leaves": 15,
            "learning_rate": 0.05,
            "min_data_in_leaf": max(5, min(20, len(y) // 10)),
            "feature_fraction": 1.0,
            "seed": 42,
            "scale_pos_weight": (len(y) - pos) / pos,
        }
        dataset = lgb.Dataset(
            np.asarray(x, dtype=float),
            label=np.asarray(y, dtype=float),
            weight=np.asarray(w, dtype=float) if w is not None else None,
        )
        booster = lgb.train(params, dataset, num_boost_round=150)
        return {
            "type": "lightgbm",
            "booster": booster.model_to_string(),
            "label_rate": pos / len(y),
        }
    except ImportError:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(x, y, sample_weight=w)
        weights = {name: float(wt) for name, wt in zip(MODEL_FEATURES, clf.coef_[0])}
        return {
            "type": "logreg",
            "weights": weights,
            "bias": float(clf.intercept_[0]),
            "label_rate": pos / len(y),
        }


def _fit_dual(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x_raw = [feature_vector(r) for r in rows]
    bounds = _clip_bounds(x_raw)
    x = _apply_clip(x_raw, bounds)
    # Uniqueness proxy for overlapping label windows: a row spanning H bars gets
    # weight 1/H so each underlying candle contributes roughly once (audit Q3).
    w = [1.0 / max(1, int(r.get("horizon_steps", 1) or 1)) for r in rows]
    y_up = [int(r.get("up_hit", 0)) for r in rows]
    y_down = [int(r.get("down_hit", 0)) for r in rows]
    pump = _fit_head(x, y_up, w)
    dump = _fit_head(x, y_down, w)
    head_types = {pump["type"], dump["type"]}
    if "lightgbm" in head_types:
        model_type = "lightgbm_dual_v1"
    elif "logreg" in head_types:
        model_type = "logreg_dual_v1"
    else:
        model_type = "constant_dual_v1"
    return {
        "model_type": model_type,
        "features": list(MODEL_FEATURES),
        "heads": {"pump": pump, "dump": dump},
        "clip_bounds": bounds,
        "train_rows": len(rows),
    }


def _predict_head(model: dict[str, Any], rows: list[dict[str, Any]], target: str) -> list[float]:
    from amber.models.infer import infer_row_prob

    return [infer_row_prob(model, r, target=target) for r in rows]


def _safe_auc(y: list[int], probs: list[float]) -> float | None:
    if len(set(y)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, probs))
    except Exception:
        return None


def _feature_quantiles(rows: list[dict[str, Any]], n_bins: int = 10) -> dict[str, list[float]]:
    """Per-feature quantile edges over the train segment — the PSI reference
    distribution used by the drift monitors."""
    out: dict[str, list[float]] = {}
    for name in MODEL_FEATURES:
        values = sorted(float(r.get(name, 0.0) or 0.0) for r in rows)
        if not values:
            continue
        edges = [values[min(len(values) - 1, int(len(values) * i / n_bins))] for i in range(n_bins + 1)]
        edges[0] = values[0]
        edges[-1] = values[-1]
        out[name] = edges
    return out


def train_model(
    datasets_root: Path,
    models_root: Path,
    *,
    cv_n_folds: int = 3,
    cv_val_size: int = 50,
    cv_gap_candles: int = 30,
    train_frac: float = 0.7,
    calib_frac: float = 0.15,
) -> dict[str, Any]:
    all_rows, dataset_run = load_latest_dataset_rows(datasets_root)
    if not all_rows:
        raise ValueError("Dataset is empty; run collectors/features/build_dataset first")

    rows, pseudo_ts, split_mode = order_with_pseudo_time(all_rows)

    val_size = max(5, int(cv_val_size))
    gap = max(1, int(cv_gap_candles))
    n_folds = max(1, int(cv_n_folds))

    splits = make_holdout_splits(pseudo_ts, train_frac=train_frac, calib_frac=calib_frac, gap=gap)
    if splits is None:
        logger.warning("dataset too small for train/calib/test holdout splits; training in-sample")
        train_rows = rows
        train_ts = pseudo_ts
    else:
        segments = split_rows(rows, pseudo_ts, splits)
        train_rows = segments["train"]
        train_ts = [t for t in pseudo_ts if t <= int(splits["train_end"])]

    folds = make_time_walk_forward_folds(train_ts, n_folds=n_folds, val_size=val_size, gap=gap)
    fold_stats: list[dict[str, float]] = []
    skipped_folds = 0
    for fold in folds:
        tr = [r for r, t in zip(train_rows, train_ts) if t <= fold["train_end"]]
        va = [r for r, t in zip(train_rows, train_ts) if fold["val_start"] <= t <= fold["val_end"]]
        if not tr or not va:
            skipped_folds += 1
            continue
        m = _fit_dual(tr)
        probs = _predict_head(m, va, target="pump")
        y = [int(r.get("up_hit", 0)) for r in va]
        brier = sum((p - yt) ** 2 for p, yt in zip(probs, y)) / len(y)
        stat: dict[str, float] = {"val_rows": float(len(va)), "brier": float(brier)}
        auc = _safe_auc(y, probs)
        if auc is not None:
            stat["auc"] = auc
        fold_stats.append(stat)

    model = _fit_dual(train_rows)
    brier_values = [float(fs["brier"]) for fs in fold_stats]
    auc_values = [float(fs["auc"]) for fs in fold_stats if "auc" in fs]
    model["cv"] = {
        "n_folds": len(fold_stats),
        "configured_n_folds": n_folds,
        "generated_folds": len(folds),
        "skipped_folds": skipped_folds,
        "val_size": val_size,
        "gap_candles": gap,
        "brier_mean": _mean(brier_values),
        "brier_std": _std(brier_values),
        "auc_mean": _mean(auc_values) if auc_values else None,
        "fold_stats": fold_stats,
    }
    model["split_mode"] = split_mode
    model["splits"] = splits
    model["train_reference"] = _feature_quantiles(train_rows)
    horizons = sorted({int(r.get("horizon_steps", 0) or 0) for r in rows if r.get("horizon_steps")})
    model["labeling"] = {
        "horizon_steps": horizons[0] if horizons else 5,
        "horizons": horizons,
        "avg_up_pct": _mean([float(r.get("up_pct", 0.0) or 0.0) for r in rows]),
        "avg_down_pct": _mean([float(r.get("down_pct", 0.0) or 0.0) for r in rows]),
    }

    run_id = new_run_id(prefix="model")
    out_dir = models_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = ArtifactManifest(
        run_id=run_id,
        artifact_type="model",
        artifact_version="v2",
        created_at=dataset_run,
        config_ref="config/amber.yaml",
        feature_spec_ref="config/features.yaml",
        metadata={
            "dataset_run": dataset_run,
            "model_type": model["model_type"],
            "train_rows": len(train_rows),
            "total_rows": len(rows),
            "split_mode": split_mode,
            "has_holdout_splits": splits is not None,
            "cv_folds": len(fold_stats),
            "cv_generated_folds": len(folds),
            "cv_skipped_folds": skipped_folds,
            "cv_gap_candles": gap,
            "cv_brier_mean": _mean(brier_values),
            "cv_brier_std": _std(brier_values),
        },
    )
    write_manifest(out_dir / "manifest.json", manifest)

    avg_raw = _mean(_predict_head(model, rows, target="pump"))
    return {
        "run_id": run_id,
        "train_rows": len(train_rows),
        "total_rows": len(rows),
        "model_type": model["model_type"],
        "split_mode": split_mode,
        "has_holdout_splits": splits is not None,
        "avg_raw_prob": avg_raw,
        "cv_folds": len(fold_stats),
        "cv_generated_folds": len(folds),
        "cv_skipped_folds": skipped_folds,
        "cv_gap_candles": gap,
        "cv_brier_mean": _mean(brier_values),
    }
