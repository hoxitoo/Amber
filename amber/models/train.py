from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from amber.common.manifest import ArtifactManifest, new_run_id, write_manifest
from amber.models.split import make_walk_forward_folds


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def _fit_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ret_vals = [float(r.get("ret_1", 0.0)) for r in rows]
    vol_vals = [float(r.get("vol_z_20", 0.0)) for r in rows]
    up_labels = [int(r.get("up_hit", 0)) for r in rows]
    down_labels = [int(r.get("down_hit", 0)) for r in rows]

    pump = {
        "weights": {"ret_1": 8.0, "vol_z_20": 0.01},
        "bias": -((sum(up_labels) / len(up_labels)) - 0.5),
        "label_rate": sum(up_labels) / len(up_labels),
    }
    dump = {
        "weights": {"ret_1": -8.0, "vol_z_20": 0.01},
        "bias": -((sum(down_labels) / len(down_labels)) - 0.5),
        "label_rate": sum(down_labels) / len(down_labels),
    }

    return {
        "model_type": "baseline_dual_logistic_v2",
        "features": ["ret_1", "vol_z_20"],
        "heads": {"pump": pump, "dump": dump},
        "train_rows": len(rows),
        "ret_mean": sum(ret_vals) / len(ret_vals),
        "vol_mean": sum(vol_vals) / len(vol_vals),
    }


def train_model(datasets_root: Path, models_root: Path) -> dict[str, Any]:
    latest = sorted([p for p in datasets_root.iterdir() if p.is_dir() and p.name.startswith("dataset_")])[-1]
    rows = _read_jsonl(latest / "dataset.jsonl")
    if not rows:
        raise ValueError("Dataset is empty; run collectors/features/build_dataset first")

    val_size = max(5, min(20, len(rows) // 4))
    gap = max(1, min(5, len(rows) // 20))
    folds = make_walk_forward_folds(len(rows), n_folds=3, val_size=val_size, gap=gap)
    fold_stats: list[dict[str, float]] = []
    for fold in folds:
        tr = rows[fold.train_start : fold.train_end]
        va = rows[fold.val_start : fold.val_end]
        if not tr or not va:
            continue
        m = _fit_baseline(tr)
        wp = m["heads"]["pump"]["weights"]
        bp = float(m["heads"]["pump"]["bias"])
        probs = [1.0 / (1.0 + math.exp(-(wp["ret_1"] * float(r.get("ret_1", 0.0)) + wp["vol_z_20"] * float(r.get("vol_z_20", 0.0)) + bp))) for r in va]
        y = [int(r.get("up_hit", 0)) for r in va]
        brier = sum((p - yt) ** 2 for p, yt in zip(probs, y)) / len(y)
        fold_stats.append({"val_rows": float(len(va)), "brier": float(brier)})

    model = _fit_baseline(rows)
    model["cv"] = {"n_folds": len(fold_stats), "fold_stats": fold_stats}

    run_id = new_run_id(prefix="model")
    out_dir = models_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = ArtifactManifest(
        run_id=run_id,
        artifact_type="model",
        artifact_version="v1",
        created_at=latest.name,
        config_ref="config/amber.yaml",
        feature_spec_ref="config/features.yaml",
        metadata={"dataset_run": latest.name, "train_rows": len(rows), "formula": "sigmoid(w_ret*ret_1+w_vol*vol_z_20+b)", "cv_folds": len(fold_stats)},
    )
    write_manifest(out_dir / "manifest.json", manifest)

    wp = model["heads"]["pump"]["weights"]
    bp = model["heads"]["pump"]["bias"]
    z = [wp["ret_1"] * float(r.get("ret_1", 0.0)) + wp["vol_z_20"] * float(r.get("vol_z_20", 0.0)) + bp for r in rows]
    p = [1.0 / (1.0 + math.exp(-v)) for v in z]
    return {"run_id": run_id, "train_rows": len(rows), "avg_raw_prob": sum(p) / len(p), "cv_folds": len(fold_stats)}
