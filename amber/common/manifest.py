from __future__ import annotations

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


def write_manifest(path: Path, manifest: ArtifactManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(manifest), fh, ensure_ascii=False, indent=2)
