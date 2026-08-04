from __future__ import annotations

import json
import math
from bisect import bisect_left
from pathlib import Path
from typing import Any
from uuid import uuid4

from amber.common.types import SignalExplanation, SignalV1
from amber.models.infer import feature_contributions, infer_row_prob, load_latest_model


def _load_latest_calibration(models_root: Path) -> dict[str, Any]:
    if not models_root.exists():
        return {"method": "identity"}
    calib_dirs = sorted([p for p in models_root.iterdir() if p.is_dir() and p.name.startswith("calib_")])
    if not calib_dirs:
        return {"method": "identity"}
    try:
        return json.loads((calib_dirs[-1] / "calibration.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"method": "identity"}


def _interp(x: list[float], y: list[float], v: float) -> float:
    if not x:
        return v
    if v <= x[0]:
        return y[0]
    if v >= x[-1]:
        return y[-1]
    i = bisect_left(x, v)
    x0, x1 = x[i - 1], x[i]
    y0, y1 = y[i - 1], y[i]
    if x1 == x0:
        return y0
    t = (v - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def calibrated_prob(raw: float, calibration: dict[str, Any]) -> float:
    method = calibration.get("method", "identity")
    if method == "platt":
        # sigmoid(a * logit(raw) + b) — continuous, so distinct raw scores stay
        # distinguishable instead of collapsing onto isotonic's few plateaus.
        eps = 1e-6
        r = min(1.0 - eps, max(eps, float(raw)))
        z = float(calibration.get("a", 1.0)) * math.log(r / (1.0 - r)) + float(calibration.get("b", 0.0))
        p = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
    elif method == "isotonic":
        p = _interp(list(calibration.get("x_thresholds", [])), list(calibration.get("y_thresholds", [])), raw)
    elif method == "scalar_mean_calibration":
        p = raw * float(calibration.get("scale", 1.0))
    else:
        p = raw
    return max(0.0, min(1.0, p))


def coherent_pump_dump(up: float, down: float) -> tuple[float, float]:
    """Jointly normalize independently-calibrated heads so (p_up, p_down, p_none)
    forms a valid distribution: when p_up + p_down > 1 both are scaled down
    proportionally, otherwise the remainder is P(no event) (audit M1)."""
    total = up + down
    if total > 1.0 and total > 0:
        return up / total, down / total
    return up, down


def calibrated_prob_for_target(raw: float, calibration: dict[str, Any], target: str) -> float:
    if calibration.get("method") == "multi_head":
        heads = calibration.get("heads", {})
        head_cal = heads.get(target) if isinstance(heads, dict) else None
        if isinstance(head_cal, dict):
            return calibrated_prob(raw, head_cal)
    return calibrated_prob(raw, calibration)




def _top_impacts(
    model: dict[str, Any],
    feature_row: dict[str, Any],
    top_n: int = 3,
) -> list[dict[str, float]]:
    contributions: dict[str, float] = {}
    for prefix, target in (("pump", "pump"), ("dump", "dump")):
        for name, value in feature_contributions(model, feature_row, target=target).items():
            contributions[f"{prefix}_{name}"] = float(value)
    ordered = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [{name: float(value)} for name, value in ordered[: max(1, top_n)]]


def score_signal(
    feature_row: dict[str, Any],
    models_root: Path,
    config_version: str = "v1",
    *,
    model: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
) -> SignalV1:
    """Score one feature row. `model`/`calibration` can be preloaded once by the
    caller (e.g. the scanner loop) to avoid re-reading artifacts per symbol."""
    if model is None:
        model = load_latest_model(models_root)
    calib = calibration if calibration is not None else _load_latest_calibration(models_root)

    up_raw = infer_row_prob(model, feature_row, target="pump")
    down_raw = infer_row_prob(model, feature_row, target="dump")

    up_cal = calibrated_prob_for_target(up_raw, calibration=calib, target="pump")
    down_cal = calibrated_prob_for_target(down_raw, calibration=calib, target="dump")
    up_cal, down_cal = coherent_pump_dump(up_cal, down_cal)

    directional = up_cal - down_cal

    explanation_method = (
        "shap_pred_contrib_top3" if model.get("model_type", "").startswith("lightgbm") else "shap_like_linear_contrib_top3"
    )
    explanation = SignalExplanation(
        top_feature_impacts=_top_impacts(model, feature_row, top_n=3),
        rule_trace=[
            {"rule": "model_inference", "passed": True},
            {"rule": "calibration_applied", "passed": True, "method": calib.get("method", "identity")},
            {"rule": "explanation_method", "value": explanation_method},
            {"rule": "directional_score", "value": directional},
        ],
    )

    # Signal metadata reflects what the model was actually trained to predict.
    labeling = model.get("labeling", {}) if isinstance(model.get("labeling"), dict) else {}
    return SignalV1(
        signal_id=f"sig_{uuid4().hex[:12]}",
        event_ts=feature_row["ts"],
        symbol=feature_row["symbol"],
        horizon_min=int(labeling.get("horizon_steps", 5)),
        target_up_pct=float(labeling.get("avg_up_pct", 0.002)),
        target_down_pct=float(labeling.get("avg_down_pct", 0.002)),
        prob_up_raw=up_raw,
        prob_down_raw=down_raw,
        prob_up_calibrated=up_cal,
        prob_down_calibrated=down_cal,
        regime="unknown",
        market_context={
            "obs": feature_row.get("obs", 0),
            "mid_price": feature_row.get("mid_price", 0.0),
            "bid": feature_row.get("bid", feature_row.get("mid_price", 0.0)),
            "ask": feature_row.get("ask", feature_row.get("mid_price", 0.0)),
            "spread_bps": feature_row.get("spread_bps", 0.0),
            "p_none": max(0.0, 1.0 - up_cal - down_cal),
        },
        explanation=explanation,
        model_version=model.get("model_type", "unknown"),
        config_version=config_version,
    )
