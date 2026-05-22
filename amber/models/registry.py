from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGISTRY_FILE = "registry.jsonl"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def register_model(models_root: Path, model_run_id: str, calibration_run_id: str | None = None) -> dict[str, Any]:
    entry = {
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "model_run_id": model_run_id,
        "calibration_run_id": calibration_run_id,
    }
    _append_jsonl(models_root / REGISTRY_FILE, entry)
    return entry


def latest_registered(models_root: Path) -> dict[str, Any] | None:
    registry = models_root / REGISTRY_FILE
    if not registry.exists():
        return None
    lines = registry.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None
<<<<<<< HEAD
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None
=======
    return json.loads(lines[-1])
>>>>>>> origin/main
