from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def load_latest_model(models_root: Path) -> dict[str, Any]:
    latest = sorted([p for p in models_root.iterdir() if p.is_dir() and p.name.startswith("model_")])[-1]
    return json.loads((latest / "model.json").read_text(encoding="utf-8"))


def infer_raw_prob(model: dict[str, Any], ret_1m: float, vol_mean: float) -> float:
    w = model["weights"]
    z = w["ret_1m"] * ret_1m + w["vol_mean"] * vol_mean + model["bias"]
    return 1.0 / (1.0 + math.exp(-z))
