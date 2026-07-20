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


def _latest_eval_metrics(logs_dir: Path) -> tuple[dict[str, float], float | None]:
    path = logs_dir / "metrics.jsonl"
    if not path.exists():
        return {}, None
    wanted = {
        "model_precision_up_at_threshold",
        "model_precision_down_at_threshold",
        "model_brier_up_cal",
        "model_brier_down_cal",
    }
    latest: dict[str, tuple[datetime, float]] = {}
    has_real_ts = False

    def _parse_ts(raw: Any) -> datetime:
        try:
            if isinstance(raw, str) and raw:
                # Accept both RFC3339 "Z" and offset forms.
                norm = raw.replace("Z", "+00:00")
                dt = datetime.fromisoformat(norm)
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
        except ValueError:
            pass
        return datetime.min.replace(tzinfo=timezone.utc)

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
            ts = _parse_ts(row.get("ts"))
            if ts != datetime.min.replace(tzinfo=timezone.utc):
                has_real_ts = True
            key = str(metric)
            prev = latest.get(key)
            if prev is None or ts >= prev[0]:
                latest[key] = (ts, value)
    metrics = {k: v for k, (_, v) in latest.items()}
    if not latest:
        return metrics, None
    if not has_real_ts:
        # Metrics are present but not timestamped; treat as freshly observed at read time.
        return metrics, 0.0
    newest_ts = max(ts for ts, _ in latest.values())
    age_sec = max(0.0, (datetime.now(timezone.utc) - newest_ts).total_seconds())
    return metrics, age_sec


def _model_eval_completeness(eval_metrics: dict[str, float]) -> dict[str, Any]:
    expected = [
        "model_precision_up_at_threshold",
        "model_precision_down_at_threshold",
        "model_brier_up_cal",
        "model_brier_down_cal",
    ]
    missing = [key for key in expected if key not in eval_metrics]
    return {
        "expected_metric_count": len(expected),
        "present_metric_count": len(expected) - len(missing),
        "missing_metrics": missing,
        "is_complete": not missing,
    }


def _model_eval_required_keys() -> tuple[str, ...]:
    return (
        "model_precision_up_at_threshold",
        "model_precision_down_at_threshold",
        "model_brier_up_cal",
        "model_brier_down_cal",
    )


def _model_eval_gate_reason(*, status: str, complete: bool) -> str:
    if status == "missing":
        return "metrics_missing"
    if status == "stale":
        return "metrics_stale"
    if not complete:
        return "metrics_incomplete"
    return "ok"


def _readiness_failed_components(
    *,
    health_ok: bool,
    backtest_ok: bool,
    model_eval_ok: bool,
    model_eval_required: bool,
) -> list[str]:
    return [
        name
        for name, failed_flag in (
            ("health", not health_ok),
            ("backtest", not backtest_ok),
            ("model_eval", (not model_eval_ok) and model_eval_required),
        )
        if failed_flag
    ]

def build_system_report(
    storage: dict[str, str],
    model_eval_fresh_sec: int = 6 * 60 * 60,
    require_model_eval_for_overall_ok: bool = True,
) -> dict[str, Any]:
    logs_dir = _require_storage_key(storage, "logs_dir")
    datasets_dir = _require_storage_key(storage, "datasets_dir")
    raw_dir = _require_storage_key(storage, "raw_dir")
    features_dir = _require_storage_key(storage, "features_dir")
    models_dir = _require_storage_key(storage, "models_dir")

    data_root = raw_dir.parent
    health = check_health(data_root=data_root, max_age_sec=15 * 60)
    quality = build_quality_report(
        logs_dir / "signals.jsonl",
        raw_root=raw_dir,
        models_root=models_dir,
        features_root=features_dir,
    )
    eval_metrics, eval_metrics_age_sec = _latest_eval_metrics(logs_dir)
    eval_completeness = _model_eval_completeness(eval_metrics)
    required_eval_metrics = _model_eval_required_keys()
    model_ready = _has_model_artifact(models_dir)

    backtest: dict[str, Any]
    backtest_ok = True
    try:
        # Prefer the model-driven replay; fall back to the label replay when no
        # trained model exists yet.
        try:
            backtest = event_backtest(datasets_dir, models_dir)
        except Exception:
            backtest = event_backtest(datasets_dir)
    except Exception as exc:  # pragma: no cover - safe runtime fallback
        backtest = {"error": str(exc)}
        backtest_ok = False

    freshness_window = max(60, int(model_eval_fresh_sec))
    if eval_metrics_age_sec is None:
        model_eval_status = "missing"
        model_eval_fresh = False
    elif eval_metrics_age_sec <= freshness_window:
        model_eval_status = "fresh"
        model_eval_fresh = True
    else:
        model_eval_status = "stale"
        model_eval_fresh = False

    model_eval_ok = model_eval_status == "fresh"
    model_eval_complete = bool(eval_completeness["is_complete"])
    model_eval_required = bool(require_model_eval_for_overall_ok)
    model_eval_gate_ok = model_eval_ok and model_eval_complete
    model_eval_gate_reason = _model_eval_gate_reason(status=model_eval_status, complete=model_eval_complete)
    overall_ok = bool(health.ok and backtest_ok and (model_eval_gate_ok if model_eval_required else True))
    overall_reason = "ok"
    model_eval_effective_failed = (not model_eval_gate_ok) and model_eval_required
    failed = {
        "health": not health.ok,
        "backtest": not backtest_ok,
        "model_eval": model_eval_effective_failed,
    }
    if all(failed.values()):
        overall_reason = "health_and_backtest_and_model_eval"
    elif failed["health"] and failed["backtest"]:
        overall_reason = "health_and_backtest"
    elif failed["health"] and failed["model_eval"]:
        overall_reason = "health_and_model_eval"
    elif failed["backtest"] and failed["model_eval"]:
        overall_reason = "backtest_and_model_eval"
    elif failed["health"]:
        overall_reason = "health"
    elif failed["backtest"]:
        overall_reason = "backtest"
    elif failed["model_eval"] and model_eval_required:
        overall_reason = "model_eval"

    readiness_failed_components = _readiness_failed_components(
        health_ok=bool(health.ok),
        backtest_ok=bool(backtest_ok),
        model_eval_ok=bool(model_eval_ok),
        model_eval_required=bool(model_eval_required),
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_ok": overall_ok,
        "overall_reason": overall_reason,
        "health": {"ok": health.ok, "checks": health.checks},
        "quality": quality,
        "model_eval": eval_metrics,
        "model_eval_age_sec": eval_metrics_age_sec,
        "model_eval_fresh": model_eval_fresh,
        "model_eval_status": model_eval_status,
        "model_eval_ok": model_eval_ok,
        "model_eval_complete": model_eval_complete,
        "model_eval_gate_ok": model_eval_gate_ok,
        "model_eval_fresh_sec": freshness_window,
        "model_eval_reasons": {
            "status": model_eval_status,
            "status_reason": (
                "metrics_missing"
                if model_eval_status == "missing"
                else ("metrics_stale" if model_eval_status == "stale" else "metrics_fresh")
            ),
            "freshness_window_sec": freshness_window,
            "age_sec": eval_metrics_age_sec,
            "has_metrics": bool(eval_metrics),
            "completeness": eval_completeness,
            "gate_ok": model_eval_gate_ok,
            "gate_reason": model_eval_gate_reason,
            "required_metrics": required_eval_metrics,
            "required_metric_count": len(required_eval_metrics),
        },
        "backtest": backtest,
        "backtest_ok": backtest_ok,
        "overall_reasons": {
            "health_failed": not health.ok,
            "backtest_failed": not backtest_ok,
            "model_eval_failed": not model_eval_ok,
            "model_eval_incomplete": not model_eval_complete,
            "model_eval_required": model_eval_required,
            "model_eval_effective_failed": model_eval_effective_failed,
            "code": overall_reason,
        },
        "readiness": {
            "health_ok": bool(health.ok),
            "backtest_ok": bool(backtest_ok),
            "model_eval_ok": bool(model_eval_ok),
            "model_eval_complete": bool(model_eval_complete),
            "model_eval_gate_ok": bool(model_eval_gate_ok),
            "overall_ok": bool(overall_ok),
        },
        "readiness_failed_components": readiness_failed_components,
        "readiness_summary": {
            "failed_count": len(readiness_failed_components),
            "has_failures": bool(readiness_failed_components),
        },
        "artifacts": {"model_ready": model_ready},
    }
