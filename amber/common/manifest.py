from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class ArtifactManifest:
    run_id: str
    artifact_type: str
    artifact_version: str
    created_at: str
    config_ref: str
    feature_spec_ref: str
    metadata: dict[str, Any]


def new_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{uuid4().hex[:8]}"


def file_sha256(path: str | Path) -> str | None:
    """Content hash of a config file, or None if it doesn't exist."""
    p = Path(path)
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def config_hashes() -> dict[str, str | None]:
    """sha256 of every config that shapes a run — ties artifacts to the exact
    config bytes, not just a path reference (audit A5)."""
    return {
        "config_sha256": file_sha256("config/amber.yaml"),
        "features_sha256": file_sha256("config/features.yaml"),
        "thresholds_sha256": file_sha256("config/thresholds.yaml"),
    }


def write_manifest(path: Path, manifest: ArtifactManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.metadata = {**config_hashes(), **manifest.metadata}
    with path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(manifest), fh, ensure_ascii=False, indent=2)
