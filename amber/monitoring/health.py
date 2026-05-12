from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class HealthReport:
    ok: bool
    checks: dict[str, Any]


def _latest_mtime(path: Path, pattern: str) -> float | None:
    files = list(path.glob(pattern))
    if not files:
        return None
    return max(f.stat().st_mtime for f in files)


def check_health(data_root: Path, max_age_sec: int = 3600) -> HealthReport:
    now = datetime.now(timezone.utc).timestamp()

    raw_mtime = _latest_mtime(data_root / "raw", "ticks/*/part-000.jsonl")
    feat_mtime = _latest_mtime(data_root / "features", "features/*/part-000.jsonl")
    model_mtime = _latest_mtime(data_root / "models", "model_*/model.json")

    checks = {
        "raw_fresh": raw_mtime is not None and (now - raw_mtime) <= max_age_sec,
        "features_fresh": feat_mtime is not None and (now - feat_mtime) <= max_age_sec,
        "model_present": model_mtime is not None,
    }
    return HealthReport(ok=all(checks.values()), checks=checks)
