from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from amber.models.dataset_io import load_latest_dataset_rows, order_with_pseudo_time, split_rows
from amber.models.infer import infer_row_prob, load_latest_model
from amber.signals.scorer import calibrated_prob_for_target

logger = logging.getLogger(__name__)


def _load_latest_calibration(models_root: Path) -> dict[str, Any]:
    calib_dirs = sorted([p for p in models_root.iterdir() if p.is_dir() and p.name.startswith("calib_")])
    if not calib_dirs:
        return {"method": "identity"}
    try:
        return json.loads((calib_dirs[-1] / "calibration.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"method": "identity"}


def _safe_auc(y: list[int], probs: list[float]) -> float | None:
    if len(set(y)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, probs))
    except Exception:
        return None


def _safe_pr_auc(y: list[int], probs: list[float]) -> float | None:
    """Average precision (PR-AUC) — the honest headline for rare events."""
    if len(set(y)) < 2:
        return None
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y, probs))
    except Exception:
        return None


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])


def regime_metrics(rows: list[dict[str, Any]], probs_up: list[float], y_up: list[int]) -> dict[str, dict[str, float]]:
    """Per-regime performance (audit Q4): trend/range × high/low volatility.

    Regimes are defined relative to the evaluation window's own medians —
    |ret_60| above median ⇒ trending, bb_width_20 above median ⇒ high vol — so
    the split is parameter-free and always yields populated buckets.
    """
    abs_ret = [abs(float(r.get("ret_60", 0.0) or 0.0)) for r in rows]
    bw = [float(r.get("bb_width_20", 0.0) or 0.0) for r in rows]
    med_ret, med_bw = _median(abs_ret), _median(bw)

    buckets: dict[str, list[int]] = {}
    for i in range(len(rows)):
        trend = "trend" if abs_ret[i] > med_ret else "range"
        vol = "hivol" if bw[i] > med_bw else "lovol"
        buckets.setdefault(f"{trend}_{vol}", []).append(i)

    out: dict[str, dict[str, float]] = {}
    for name, idx in sorted(buckets.items()):
        y = [y_up[i] for i in idx]
        p = [probs_up[i] for i in idx]
        base = sum(y) / len(y) if y else 0.0
        metrics: dict[str, float] = {
            "rows": float(len(idx)),
            "base_rate_up": base,
            "brier_up": sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(y) if y else 0.0,
        }
        pr = _safe_pr_auc(y, p)
        if pr is not None:
            metrics["pr_auc_up"] = pr
            if base > 0:
                metrics["pr_auc_up_lift"] = pr / base
        out[name] = metrics
    return out


def _reliability_curve(y: list[int], probs: list[float], bins: int = 10) -> list[dict[str, float]]:
    """Binned calibration: mean predicted prob vs observed frequency per bin."""
    buckets: list[dict[str, float]] = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (p >= lo and (p < hi or (b == bins - 1 and p <= hi)))]
        if not idx:
            continue
        pred = sum(probs[i] for i in idx) / len(idx)
        obs = sum(y[i] for i in idx) / len(idx)
        buckets.append({"bin": b, "n": len(idx), "mean_pred": pred, "observed": obs})
    return buckets


def evaluate_model(models_root: Path, datasets_root: Path, threshold: float = 0.7) -> dict[str, float]:
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")
    model = load_latest_model(models_root)
    calibration = _load_latest_calibration(models_root)
    all_rows, _dataset_run = load_latest_dataset_rows(datasets_root)
    if not all_rows:
        raise ValueError("Dataset is empty; cannot evaluate")

    ordered, pseudo_ts, _mode = order_with_pseudo_time(all_rows)
    splits = model.get("splits")
    in_sample = True
    rows = ordered
    if isinstance(splits, dict):
        test_rows = split_rows(ordered, pseudo_ts, splits)["test"]
        if test_rows:
            rows = test_rows
            in_sample = False
        else:
            logger.warning("model splits produced empty test segment; evaluating in-sample")
    else:
        logger.warning("model has no holdout splits; evaluation metrics are in-sample")

    probs_up_raw = [infer_row_prob(model, r, target="pump") for r in rows]
    probs_down_raw = [infer_row_prob(model, r, target="dump") for r in rows]
    probs_up_cal = [calibrated_prob_for_target(p, calibration, target="pump") for p in probs_up_raw]
    probs_down_cal = [calibrated_prob_for_target(p, calibration, target="dump") for p in probs_down_raw]

    y_up = [int(r.get("up_hit", 0)) for r in rows]
    y_down = [int(r.get("down_hit", 0)) for r in rows]

    preds_up = [1 if p >= threshold else 0 for p in probs_up_cal]
    tp_up = sum(1 for yp, yt in zip(preds_up, y_up) if yp == 1 and yt == 1)
    fp_up = sum(1 for yp, yt in zip(preds_up, y_up) if yp == 1 and yt == 0)
    precision_up = 0.0 if tp_up + fp_up == 0 else tp_up / (tp_up + fp_up)

    preds_down = [1 if p >= threshold else 0 for p in probs_down_cal]
    tp_down = sum(1 for yp, yt in zip(preds_down, y_down) if yp == 1 and yt == 1)
    fp_down = sum(1 for yp, yt in zip(preds_down, y_down) if yp == 1 and yt == 0)
    precision_down = 0.0 if tp_down + fp_down == 0 else tp_down / (tp_down + fp_down)
    brier_up_cal = sum((p - yt) ** 2 for p, yt in zip(probs_up_cal, y_up)) / len(y_up)
    brier_down_cal = sum((p - yt) ** 2 for p, yt in zip(probs_down_cal, y_down)) / len(y_down)
    brier = 0.5 * (brier_up_cal + brier_down_cal)

    out: dict[str, float] = {
        "rows": float(len(y_up)),
        "rows_total": float(len(ordered)),
        "in_sample": 1.0 if in_sample else 0.0,
        "precision_at_threshold": precision_up,
        "precision_up_at_threshold": precision_up,
        "precision_down_at_threshold": precision_down,
        "brier": brier,
        "avg_prob": sum(probs_up_cal) / len(probs_up_cal),
        "avg_prob_down": sum(probs_down_cal) / len(probs_down_cal),
        "brier_up_cal": brier_up_cal,
        "brier_down_cal": brier_down_cal,
        "base_rate_up": sum(y_up) / len(y_up),
        "base_rate_down": sum(y_down) / len(y_down),
        "calibration_method": str(calibration.get("method", "identity")),  # type: ignore[dict-item]
    }
    auc_up = _safe_auc(y_up, probs_up_cal)
    if auc_up is not None:
        out["auc_up_cal"] = auc_up
    auc_down = _safe_auc(y_down, probs_down_cal)
    if auc_down is not None:
        out["auc_down_cal"] = auc_down

    # PR-AUC / average precision — headline metric under class imbalance.
    pr_up = _safe_pr_auc(y_up, probs_up_cal)
    if pr_up is not None:
        out["pr_auc_up_cal"] = pr_up
    pr_down = _safe_pr_auc(y_down, probs_down_cal)
    if pr_down is not None:
        out["pr_auc_down_cal"] = pr_down
    # PR-AUC lift over the base rate (>1 means better than always-predict-positive).
    if pr_up is not None and out["base_rate_up"] > 0:
        out["pr_auc_up_lift"] = pr_up / out["base_rate_up"]
    if pr_down is not None and out["base_rate_down"] > 0:
        out["pr_auc_down_lift"] = pr_down / out["base_rate_down"]
    return out


def evaluate_model_verbose(models_root: Path, datasets_root: Path, threshold: float = 0.7) -> dict[str, Any]:
    """Scalar metrics plus the reliability (calibration) curves for dashboards."""
    metrics = evaluate_model(models_root, datasets_root, threshold=threshold)
    model = load_latest_model(models_root)
    calibration = _load_latest_calibration(models_root)
    all_rows, _ = load_latest_dataset_rows(datasets_root)
    ordered, pseudo_ts, _mode = order_with_pseudo_time(all_rows)
    splits = model.get("splits")
    rows = ordered
    if isinstance(splits, dict):
        test_rows = split_rows(ordered, pseudo_ts, splits)["test"]
        if test_rows:
            rows = test_rows
    p_up = [calibrated_prob_for_target(infer_row_prob(model, r, target="pump"), calibration, target="pump") for r in rows]
    y_up = [int(r.get("up_hit", 0)) for r in rows]
    return {
        **metrics,
        "reliability_up": _reliability_curve(y_up, p_up),
        "regimes": regime_metrics(rows, p_up, y_up),
    }
