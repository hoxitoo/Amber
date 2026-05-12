from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from amber.common.types import SignalExplanation, SignalV1
from amber.models.infer import infer_raw_prob, load_latest_model


def _load_latest_calibration(models_root: Path) -> dict[str, Any]:
    calib_dirs = sorted([p for p in models_root.iterdir() if p.is_dir() and p.name.startswith("calib_")])
    if not calib_dirs:
        return {"scale": 1.0, "method": "identity"}
    return json.loads((calib_dirs[-1] / "calibration.json").read_text(encoding="utf-8"))


def calibrated_prob(raw: float, scale: float) -> float:
    p = raw * scale
    if p < 0:
        return 0.0
    if p > 1:
        return 1.0
    return p


def score_signal(feature_row: dict[str, Any], models_root: Path, config_version: str = "v1") -> SignalV1:
    model = load_latest_model(models_root)
    calib = _load_latest_calibration(models_root)

    ret_1m = float(feature_row.get("ret_1m", 0.0))
    vol_mean = float(feature_row.get("vol_mean", 0.0))

    up_raw = infer_raw_prob(model, ret_1m=ret_1m, vol_mean=vol_mean)
    down_raw = max(0.0, min(1.0, 1.0 - up_raw))
    scale = float(calib.get("scale", 1.0))

    up_cal = calibrated_prob(up_raw, scale=scale)
    down_cal = calibrated_prob(down_raw, scale=scale)

    explanation = SignalExplanation(
        top_feature_impacts=[
            {"ret_1m": float(model["weights"]["ret_1m"]) * ret_1m},
            {"vol_mean": float(model["weights"]["vol_mean"]) * vol_mean},
        ],
        rule_trace=[
            {"rule": "model_inference", "passed": True},
            {"rule": "calibration_applied", "passed": True, "method": calib.get("method", "identity")},
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
        market_context={"obs": feature_row.get("obs", 0)},
        explanation=explanation,
        model_version=model.get("model_type", "unknown"),
        config_version=config_version,
    )
