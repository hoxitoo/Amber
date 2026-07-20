from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from amber.common.manifest import ArtifactManifest, new_run_id, write_manifest
from amber.models.dataset_io import load_latest_dataset_rows, order_with_pseudo_time, split_rows
from amber.models.infer import infer_row_prob, load_latest_model
from amber.models.registry import latest_registered

logger = logging.getLogger(__name__)


def _select_holdout(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    pseudo_ts: list[int],
    holdout_ratio: float,
) -> tuple[list[dict[str, Any]], str]:
    """Prefer the model's dedicated calibration segment (out-of-sample for the
    trained model); fall back to a tail slice for legacy/small datasets."""
    splits = model.get("splits")
    if isinstance(splits, dict):
        holdout = split_rows(rows, pseudo_ts, splits)["calib"]
        if holdout:
            return holdout, "holdout_split"
        logger.warning("model splits produced empty calibration segment; falling back to tail slice")
    holdout_size = max(20, int(len(rows) * holdout_ratio))
    holdout_size = min(holdout_size, len(rows))
    return rows[-holdout_size:], "tail_in_sample"


def calibrate_model(models_root: Path, datasets_root: Path, holdout_ratio: float = 0.2) -> dict[str, Any]:
    model = load_latest_model(models_root)
    reg = latest_registered(models_root)
    model_run_id = reg["model_run_id"] if reg else "unregistered"

    all_rows, dataset_run = load_latest_dataset_rows(datasets_root)
    if not all_rows:
        raise ValueError("Dataset is empty; cannot calibrate")
    if holdout_ratio <= 0.0 or holdout_ratio >= 1.0:
        raise ValueError("holdout_ratio must be in (0, 1)")

    rows, pseudo_ts, _mode = order_with_pseudo_time(all_rows)
    holdout, holdout_source = _select_holdout(model, rows, pseudo_ts, holdout_ratio)

    def _fit_head(target: str, label_key: str) -> dict[str, Any]:
        raw = [infer_row_prob(model, r, target=target) for r in holdout]
        y = [int(r.get(label_key, 0)) for r in holdout]
        try:
            from sklearn.isotonic import IsotonicRegression  # type: ignore

            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw, y)
            return {
                "method": "isotonic",
                "x_thresholds": [float(v) for v in iso.X_thresholds_],  # type: ignore[attr-defined]
                "y_thresholds": [float(v) for v in iso.y_thresholds_],  # type: ignore[attr-defined]
            }
        except Exception:
            mean_raw = sum(raw) / len(raw)
            observed = sum(y) / len(y)
            scale = 1.0 if mean_raw == 0 else observed / mean_raw
            scale = max(0.1, min(2.0, scale))
            return {
                "method": "scalar_mean_calibration",
                "scale": scale,
                "mean_raw": mean_raw,
                "observed": observed,
            }

    pump_cal = _fit_head(target="pump", label_key="up_hit")
    dump_cal = _fit_head(target="dump", label_key="down_hit")
    payload: dict[str, Any] = {
        "method": "multi_head",
        "heads": {"pump": pump_cal, "dump": dump_cal},
        "model_run_id": model_run_id,
        "rows": len(holdout),
        "holdout_ratio": holdout_ratio,
        "holdout_source": holdout_source,
    }

    calib_run = new_run_id(prefix="calib")
    out_dir = models_root / calib_run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = ArtifactManifest(
        run_id=calib_run,
        artifact_type="calibration",
        artifact_version="v2",
        created_at=dataset_run,
        config_ref="config/amber.yaml",
        feature_spec_ref="config/features.yaml",
        metadata={
            "dataset_run": dataset_run,
            "rows": len(holdout),
            "model_type": model.get("model_type"),
            "model_run_id": model_run_id,
            "method": payload["method"],
            "pump_method": pump_cal["method"],
            "dump_method": dump_cal["method"],
            "holdout_ratio": holdout_ratio,
            "holdout_source": holdout_source,
        },
    )
    write_manifest(out_dir / "manifest.json", manifest)
    return {
        "run_id": calib_run,
        "method": payload["method"],
        "model_run_id": model_run_id,
        "holdout_rows": len(holdout),
        "holdout_source": holdout_source,
    }
