from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from amber.models.registry import latest_registered


def load_latest_model(models_root: Path) -> dict[str, Any]:
    reg = latest_registered(models_root)
    if reg:
        model_dir = models_root / reg["model_run_id"]
        return json.loads((model_dir / "model.json").read_text(encoding="utf-8"))

    latest = sorted([p for p in models_root.iterdir() if p.is_dir() and p.name.startswith("model_")])[-1]
    return json.loads((latest / "model.json").read_text(encoding="utf-8"))


def infer_raw_prob(model: dict[str, Any], ret_1: float, vol_z_20: float) -> float:
    w = model["weights"]
    z = w["ret_1"] * ret_1 + w["vol_z_20"] * vol_z_20 + model["bias"]
    return 1.0 / (1.0 + math.exp(-z))
