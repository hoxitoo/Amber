from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from amber.models.infer import infer_raw_prob, load_latest_model
from amber.signals.scorer import calibrated_prob_for_target


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {lineno}: {exc.msg}") from exc
    return rows


def _load_latest_calibration(models_root: Path) -> dict[str, Any]:
    calib_dirs = sorted([p for p in models_root.iterdir() if p.is_dir() and p.name.startswith("calib_")])
    if not calib_dirs:
        return {"method": "identity"}
    try:
        return json.loads((calib_dirs[-1] / "calibration.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"method": "identity"}


def evaluate_model(models_root: Path, datasets_root: Path, threshold: float = 0.7) -> dict[str, float]:
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")
    model = load_latest_model(models_root)
    calibration = _load_latest_calibration(models_root)
    if not datasets_root.exists():
        raise ValueError(f"No dataset_* directories found under: {datasets_root}")
    candidates = sorted([p for p in datasets_root.iterdir() if p.is_dir() and p.name.startswith("dataset_")])
    if not candidates:
        raise ValueError(f"No dataset_* directories found under: {datasets_root}")
    latest_ds = candidates[-1]
    rows = _read_jsonl(latest_ds / "dataset.jsonl")
    if not rows:
        raise ValueError("Dataset is empty; cannot evaluate")

    probs_up_raw = [infer_raw_prob(model, float(r.get("ret_1", 0.0)), float(r.get("vol_z_20", 0.0)), target="pump") for r in rows]
    probs_down_raw = [infer_raw_prob(model, float(r.get("ret_1", 0.0)), float(r.get("vol_z_20", 0.0)), target="dump") for r in rows]
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

    return {
        "rows": float(len(y_up)),
        "precision_at_threshold": precision_up,
        "precision_up_at_threshold": precision_up,
        "precision_down_at_threshold": precision_down,
        "brier": brier,
        "avg_prob": sum(probs_up_cal) / len(probs_up_cal),
        "avg_prob_down": sum(probs_down_cal) / len(probs_down_cal),
        "brier_up_cal": brier_up_cal,
        "brier_down_cal": brier_down_cal,
        "calibration_method": str(calibration.get("method", "identity")),
    }
