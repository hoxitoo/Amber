from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from amber.common.manifest import ArtifactManifest, new_run_id, write_manifest
from amber.models.infer import infer_raw_prob, load_latest_model
from amber.models.registry import latest_registered


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {lineno}: {exc.msg}") from exc
    return rows


def calibrate_model(models_root: Path, datasets_root: Path) -> dict[str, Any]:
    model = load_latest_model(models_root)
    reg = latest_registered(models_root)
    model_run_id = reg["model_run_id"] if reg else "unregistered"

    if not datasets_root.exists():
        raise ValueError(f"No dataset_* directories found under: {datasets_root}")
    candidates = sorted([p for p in datasets_root.iterdir() if p.is_dir() and p.name.startswith("dataset_")])
    if not candidates:
        raise ValueError(f"No dataset_* directories found under: {datasets_root}")
    latest_ds = candidates[-1]
    rows = _read_jsonl(latest_ds / "dataset.jsonl")
    if not rows:
        raise ValueError("Dataset is empty; cannot calibrate")

    raw = [infer_raw_prob(model, float(r.get("ret_1", 0.0)), float(r.get("vol_z_20", 0.0)), target="pump") for r in rows]
    y = [int(r.get("up_hit", 0)) for r in rows]

    payload: dict[str, Any]
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw, y)
        x = [float(v) for v in iso.X_thresholds_]  # type: ignore[attr-defined]
        yhat = [float(v) for v in iso.y_thresholds_]  # type: ignore[attr-defined]
        payload = {
            "method": "isotonic",
            "x_thresholds": x,
            "y_thresholds": yhat,
            "model_run_id": model_run_id,
            "rows": len(rows),
        }
    except Exception:
        mean_raw = sum(raw) / len(raw)
        observed = sum(y) / len(y)
        scale = 1.0 if mean_raw == 0 else observed / mean_raw
        scale = max(0.1, min(2.0, scale))
        payload = {
            "method": "scalar_mean_calibration",
            "scale": scale,
            "mean_raw": mean_raw,
            "observed": observed,
            "model_run_id": model_run_id,
            "rows": len(rows),
        }

    calib_run = new_run_id(prefix="calib")
    out_dir = models_root / calib_run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = ArtifactManifest(
        run_id=calib_run,
        artifact_type="calibration",
        artifact_version="v1",
        created_at=latest_ds.name,
        config_ref="config/amber.yaml",
        feature_spec_ref="config/features.yaml",
        metadata={"dataset_run": latest_ds.name, "rows": len(rows), "model_type": model.get("model_type"), "model_run_id": model_run_id, "method": payload["method"]},
    )
    write_manifest(out_dir / "manifest.json", manifest)
    return {"run_id": calib_run, "method": payload["method"], "model_run_id": model_run_id}
