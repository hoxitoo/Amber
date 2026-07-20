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
    latest: float | None = None
    for file in path.glob(pattern):
        mtime = file.stat().st_mtime
        latest = mtime if latest is None else max(latest, mtime)
    return latest


def check_health(data_root: Path, max_age_sec: int = 3600) -> HealthReport:
    if max_age_sec <= 0:
        raise ValueError("max_age_sec must be positive")
    now = datetime.now(timezone.utc).timestamp()

    # `normalized` is the topic every collector/normalizer actually writes.
    raw_mtime = _latest_mtime(data_root / "raw", "normalized/*/part-*.jsonl")
    feat_mtime = _latest_mtime(data_root / "features", "features/*/part-*.jsonl")
    model_mtime = _latest_mtime(data_root / "models", "model_*/model.json")

    raw_age_sec = None if raw_mtime is None else now - raw_mtime
    feat_age_sec = None if feat_mtime is None else now - feat_mtime
    model_age_sec = None if model_mtime is None else now - model_mtime
    raw_fresh = raw_age_sec is not None and raw_age_sec <= max_age_sec
    features_fresh = feat_age_sec is not None and feat_age_sec <= max_age_sec
    model_present = model_mtime is not None
    checks = {
        "raw_fresh": raw_fresh,
        "features_fresh": features_fresh,
        "model_present": model_present,
        "raw_age_sec": raw_age_sec,
        "features_age_sec": feat_age_sec,
        "model_age_sec": model_age_sec,
    }
    return HealthReport(ok=raw_fresh and features_fresh and model_present, checks=checks)
