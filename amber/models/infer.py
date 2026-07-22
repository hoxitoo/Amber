from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from amber.models.features import MODEL_FEATURES, feature_vector
from amber.models.registry import latest_registered


def load_latest_model(models_root: Path) -> dict[str, Any]:
    reg = latest_registered(models_root)
    if reg:
        model_path = models_root / reg["model_run_id"] / "model.json"
        if model_path.exists():
            return json.loads(model_path.read_text(encoding="utf-8"))

    if not models_root.exists():
        raise ValueError(f"No model_* directories found under: {models_root}")
    candidates = sorted([p for p in models_root.iterdir() if p.is_dir() and p.name.startswith("model_")])
    if not candidates:
        raise ValueError(f"No model_* directories found under: {models_root}")
    latest = candidates[-1]
    return json.loads((latest / "model.json").read_text(encoding="utf-8"))


def _resolve_head(model: dict[str, Any], target: str) -> dict[str, Any]:
    heads = model.get("heads")
    if isinstance(heads, dict) and target in heads:
        return heads[target]
    return {"weights": model["weights"], "bias": model["bias"]}


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _lgb_booster(head: dict[str, Any]) -> Any:
    booster = head.get("_booster_obj")
    if booster is None:
        import lightgbm as lgb

        booster = lgb.Booster(model_str=head["booster"])
        head["_booster_obj"] = booster
    return booster


def _clip_value(model: dict[str, Any], name: str, value: float) -> float:
    bounds = model.get("clip_bounds")
    if isinstance(bounds, dict) and name in bounds:
        lo, hi = bounds[name]
        return min(float(hi), max(float(lo), value))
    return value


def _clipped_vector(model: dict[str, Any], feature_row: dict[str, Any]) -> list[float]:
    """Feature vector with the train-time winsorization bounds applied, so
    inference sees the same value ranges the model was fitted on (audit M4)."""
    names = list(model.get("features") or MODEL_FEATURES)
    vec = feature_vector(feature_row, names)
    return [_clip_value(model, name, v) for name, v in zip(names, vec)]


def _linear_logit(model: dict[str, Any], head: dict[str, Any], feature_row: dict[str, Any]) -> float:
    w = head.get("weights", {})
    z = float(head.get("bias", 0.0))
    for name, weight in w.items():
        z += float(weight) * _clip_value(model, name, float(feature_row.get(name, 0.0) or 0.0))
    return z


def infer_row_prob(model: dict[str, Any], feature_row: dict[str, Any], target: str = "pump") -> float:
    """Raw event probability for a full feature row from any supported head type."""
    head = _resolve_head(model, target=target)
    head_type = head.get("type")
    if head_type == "constant":
        return min(1.0, max(0.0, float(head["prob"])))
    if head_type == "lightgbm":
        vec = _clipped_vector(model, feature_row)
        p = float(_lgb_booster(head).predict([vec])[0])
        return min(1.0, max(0.0, p))
    # "logreg" and legacy linear heads share the same formula
    return _sigmoid(_linear_logit(model, head, feature_row))


def infer_raw_prob(model: dict[str, Any], ret_1: float, vol_z_20: float, target: str = "pump") -> float:
    """Legacy 2-feature entry point; prefer `infer_row_prob` with a full row."""
    return infer_row_prob(model, {"ret_1": ret_1, "vol_z_20": vol_z_20}, target=target)


def feature_contributions(model: dict[str, Any], feature_row: dict[str, Any], target: str = "pump") -> dict[str, float]:
    """Per-feature contribution to the head's score (SHAP values for LightGBM,
    weight*value for linear heads). Used for signal explanations."""
    head = _resolve_head(model, target=target)
    head_type = head.get("type")
    if head_type == "constant":
        return {}
    if head_type == "lightgbm":
        names = list(model.get("features") or MODEL_FEATURES)
        vec = _clipped_vector(model, feature_row)
        contribs = _lgb_booster(head).predict([vec], pred_contrib=True)[0]
        return {name: float(c) for name, c in zip(names, contribs)}
    w = head.get("weights", {})
    return {
        name: float(weight) * _clip_value(model, name, float(feature_row.get(name, 0.0) or 0.0))
        for name, weight in w.items()
    }
