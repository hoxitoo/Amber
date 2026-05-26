from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from amber.backtest.backtester import event_backtest
from amber.monitoring.health import check_health
from amber.monitoring.quality_report import build_quality_report




def _require_storage_key(storage: dict[str, str], key: str) -> Path:
    value = storage.get(key)
    if not value:
        raise ValueError(f"Missing storage key: {key}")
    return Path(value)




def _has_model_artifact(models_dir: Path) -> bool:
    if not models_dir.exists():
        return False
    for candidate in models_dir.glob("**/model.json"):
        if candidate.is_file():
            return True
    return False


def _latest_eval_metrics(logs_dir: Path) -> dict[str, float]:
    path = logs_dir / "metrics.jsonl"
    if not path.exists():
        return {}
    wanted = {
        "model_precision_up_at_threshold",
        "model_precision_down_at_threshold",
        "model_brier_up_cal",
        "model_brier_down_cal",
    }
    latest: dict[str, tuple[str, float]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            metric = row.get("metric")
            if metric not in wanted:
                continue
            try:
                value = float(row.get("value"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            ts = str(row.get("ts", ""))
            key = str(metric)
            prev = latest.get(key)
            if prev is None or ts >= prev[0]:
                latest[key] = (ts, value)
    return {k: v for k, (_, v) in latest.items()}

def build_system_report(storage: dict[str, str]) -> dict[str, Any]:
    logs_dir = _require_storage_key(storage, "logs_dir")
    datasets_dir = _require_storage_key(storage, "datasets_dir")
    raw_dir = _require_storage_key(storage, "raw_dir")
    features_dir = _require_storage_key(storage, "features_dir")
    models_dir = _require_storage_key(storage, "models_dir")

    data_root = raw_dir.parent
    health = check_health(data_root=data_root, max_age_sec=15 * 60)
    quality = build_quality_report(logs_dir / "signals.jsonl")
    eval_metrics = _latest_eval_metrics(logs_dir)
    model_ready = _has_model_artifact(models_dir)

    backtest: dict[str, Any]
    backtest_ok = True
    try:
        backtest = event_backtest(datasets_dir)
    except Exception as exc:  # pragma: no cover - safe runtime fallback
        backtest = {"error": str(exc)}
        backtest_ok = False

    overall_ok = bool(health.ok and backtest_ok)
    overall_reason = "ok"
    if not health.ok and not backtest_ok:
        overall_reason = "health_and_backtest"
    elif not health.ok:
        overall_reason = "health"
    elif not backtest_ok:
        overall_reason = "backtest"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_ok": overall_ok,
        "overall_reason": overall_reason,
        "health": {"ok": health.ok, "checks": health.checks},
        "quality": quality,
        "model_eval": eval_metrics,
        "backtest": backtest,
        "backtest_ok": backtest_ok,
        "artifacts": {"model_ready": model_ready},
    }
