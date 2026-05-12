from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from amber.models.infer import infer_raw_prob, load_latest_model


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def evaluate_model(models_root: Path, datasets_root: Path, threshold: float = 0.7) -> dict[str, float]:
    model = load_latest_model(models_root)
    latest_ds = sorted([p for p in datasets_root.iterdir() if p.is_dir() and p.name.startswith("dataset_")])[-1]
    rows = _read_jsonl(latest_ds / "dataset.jsonl")

    probs = [infer_raw_prob(model, float(r.get("ret_1", 0.0)), float(r.get("vol_z_20", 0.0))) for r in rows]
    y = [int(r.get("up_hit", 0)) for r in rows]

    preds = [1 if p >= threshold else 0 for p in probs]
    tp = sum(1 for yp, yt in zip(preds, y) if yp == 1 and yt == 1)
    fp = sum(1 for yp, yt in zip(preds, y) if yp == 1 and yt == 0)
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    brier = sum((p - yt) ** 2 for p, yt in zip(probs, y)) / len(y)

    return {"rows": float(len(y)), "precision_at_threshold": precision, "brier": brier, "avg_prob": sum(probs) / len(probs)}
