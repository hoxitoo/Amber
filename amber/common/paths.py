from __future__ import annotations

from pathlib import Path


def artifact_dir(root: Path, kind: str, run_id: str) -> Path:
    target = root / kind / run_id
    target.mkdir(parents=True, exist_ok=True)
    return target
