from __future__ import annotations

from pathlib import Path
import re


def artifact_dir(root: Path, kind: str, run_id: str) -> Path:
    root = root.resolve()
    _validate_segment(kind, "kind")
    _validate_segment(run_id, "run_id")
    target = (root / kind / run_id).resolve()
    if root not in target.parents and target != root:
        raise ValueError("Artifact path escapes root")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _validate_segment(value: str, name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"Invalid {name}: {value!r}")
