from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path
from typing import Any
from uuid import uuid4

from amber.common.types import SignalExplanation, SignalV1
from amber.models.infer import infer_raw_prob, load_latest_model


def _load_latest_calibration(models_root: Path) -> dict[str, Any]:
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
    if method == "isotonic":
        p = _interp(list(calibration.get("x_thresholds", [])), list(calibration.get("y_thresholds", [])), raw)
    elif method == "scalar_mean_calibration":
        p = raw * float(calibration.get("scale", 1.0))
    else:
        p = raw
    return max(0.0, min(1.0, p))


def calibrated_prob_for_target(raw: float, calibration: dict[str, Any], target: str) -> float:
    if calibration.get("method") == "multi_head":
        heads = calibration.get("heads", {})
        head_cal = heads.get(target) if isinstance(heads, dict) else None
        if isinstance(head_cal, dict):
            return calibrated_prob(raw, head_cal)
    return calibrated_prob(raw, calibration)




def _head_weights(model: dict[str, Any], target: str) -> dict[str, float]:
    heads = model.get("heads", {})
    if isinstance(heads, dict) and target in heads:
        w = heads[target].get("weights", {})
        return {"ret_1": float(w.get("ret_1", 0.0)), "vol_z_20": float(w.get("vol_z_20", 0.0))}
    w = model.get("weights", {})
    return {"ret_1": float(w.get("ret_1", 0.0)), "vol_z_20": float(w.get("vol_z_20", 0.0))}


def _top_shap_like_impacts(
    pump_w: dict[str, float],
    dump_w: dict[str, float],
    ret_1: float,
    vol_z_20: float,
    top_n: int = 3,
) -> list[dict[str, float]]:
    contributions = {
        "pump_ret_1": pump_w["ret_1"] * ret_1,
        "pump_vol_z_20": pump_w["vol_z_20"] * vol_z_20,
        "dump_ret_1": dump_w["ret_1"] * ret_1,
        "dump_vol_z_20": dump_w["vol_z_20"] * vol_z_20,
    }
    ordered = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [{name: float(value)} for name, value in ordered[: max(1, top_n)]]


def score_signal(feature_row: dict[str, Any], models_root: Path, config_version: str = "v1") -> SignalV1:
    model = load_latest_model(models_root)
    calib = _load_latest_calibration(models_root)

    ret_1 = float(feature_row.get("ret_1", 0.0))
    vol_z_20 = float(feature_row.get("vol_z_20", 0.0))

    up_raw = infer_raw_prob(model, ret_1=ret_1, vol_z_20=vol_z_20, target="pump")
    down_raw = infer_raw_prob(model, ret_1=ret_1, vol_z_20=vol_z_20, target="dump")

    up_cal = calibrated_prob_for_target(up_raw, calibration=calib, target="pump")
    down_cal = calibrated_prob_for_target(down_raw, calibration=calib, target="dump")

    pump_w = _head_weights(model, target="pump")
    dump_w = _head_weights(model, target="dump")
    directional = up_cal - down_cal

    explanation = SignalExplanation(
        top_feature_impacts=_top_shap_like_impacts(pump_w, dump_w, ret_1=ret_1, vol_z_20=vol_z_20, top_n=3),
        rule_trace=[
            {"rule": "model_inference", "passed": True},
            {"rule": "calibration_applied", "passed": True, "method": calib.get("method", "identity")},
            {"rule": "explanation_method", "value": "shap_like_linear_contrib_top3"},
            {"rule": "directional_score", "value": directional},
        ],
    )

    return SignalV1(
        signal_id=f"sig_{uuid4().hex[:12]}",
        event_ts=feature_row["ts"],
        symbol=feature_row["symbol"],
        horizon_min=5,
        target_up_pct=0.2,
        target_down_pct=0.2,
        prob_up_raw=up_raw,
        prob_down_raw=down_raw,
        prob_up_calibrated=up_cal,
        prob_down_calibrated=down_cal,
        regime="unknown",
        market_context={"obs": feature_row.get("obs", 0), "mid_price": feature_row.get("mid_price", 0.0), "bid": feature_row.get("bid", feature_row.get("mid_price", 0.0)), "ask": feature_row.get("ask", feature_row.get("mid_price", 0.0)), "spread_bps": feature_row.get("spread_bps", 0.0)},
        explanation=explanation,
        model_version=model.get("model_type", "unknown"),
        config_version=config_version,
    )
