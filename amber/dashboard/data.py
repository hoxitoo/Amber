"""Data-loading helpers for the Amber dashboard.

Everything here is read-only and reuses the existing library functions so the
dashboard never diverges from what the pipeline actually produces.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root by walking up until config/amber.yaml is found."""
    candidates = [Path.cwd()]
    here = (start or Path(__file__)).resolve()
    candidates.extend(here.parents)
    for p in candidates:
        if (p / "config" / "amber.yaml").is_file():
            return p
    return Path.cwd()


def load_config(project_root: Path) -> dict[str, Any]:
    from amber.common.config import ConfigLoader

    return ConfigLoader(project_root).load_yaml("config/amber.yaml")


def load_thresholds(project_root: Path) -> dict[str, Any]:
    from amber.common.config import ConfigLoader

    return ConfigLoader(project_root).load_yaml("config/thresholds.yaml").get("thresholds", {})


def storage_paths(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Absolute storage paths (config values are relative to the project root).

    Non-path entries in the storage section (e.g. `keep_runs: 5`) are passed
    through untouched.
    """
    out: dict[str, Any] = {}
    for key, value in config.get("storage", {}).items():
        if not isinstance(value, str):
            out[key] = value
            continue
        p = Path(value)
        out[key] = str(p if p.is_absolute() else project_root / p)
    return out


def safe_system_report(
    storage: dict[str, str],
    *,
    model_eval_fresh_sec: int = 21600,
    require_model_eval: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    from amber.monitoring.reporting import build_system_report

    try:
        report = build_system_report(
            storage,
            model_eval_fresh_sec=model_eval_fresh_sec,
            require_model_eval_for_overall_ok=require_model_eval,
            # Never run the backtest replay inside the UI process — it loads the
            # whole dataset per render and OOM-kills the dashboard. Show the
            # result published by the hourly retrain instead.
            compute_backtest=False,
        )
        return report, None
    except Exception as exc:  # dashboard must never crash on partial data
        return None, str(exc)


def _read_jsonl_tail(path: Path, max_lines: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    # Stream with a bounded deque instead of read_text(): signals.jsonl and the
    # normalized parts grow without limit, and slurping them into the UI process
    # is a steady memory leak across renders.
    with path.open("r", encoding="utf-8") as fh:
        lines = deque(fh, maxlen=max_lines)
    out: list[dict[str, Any]] = []
    for raw in lines:
        if not raw.strip():
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return out


def load_signals(logs_dir: str, limit: int = 100) -> list[dict[str, Any]]:
    """Most recent signals first."""
    rows = _read_jsonl_tail(Path(logs_dir) / "signals.jsonl", max_lines=limit * 3)
    return rows[-limit:][::-1]


def load_latest_model(models_dir: str) -> dict[str, Any] | None:
    try:
        from amber.models.infer import load_latest_model as _load

        return _load(Path(models_dir))
    except Exception:
        return None


def dataset_info(datasets_dir: str) -> dict[str, Any] | None:
    try:
        from amber.models.dataset_io import latest_dataset_dir

        d = latest_dataset_dir(Path(datasets_dir))
    except Exception:
        return None
    info: dict[str, Any] = {"run_id": d.name}
    manifest = d / "manifest.json"
    if manifest.exists():
        try:
            meta = json.loads(manifest.read_text(encoding="utf-8")).get("metadata", {})
            info.update(
                {
                    "rows": meta.get("rows"),
                    "symbols": meta.get("symbols", []),
                    "horizons": meta.get("horizons", []),
                }
            )
        except json.JSONDecodeError:
            pass

    # The dataset is a rolling window, so its row count is constant by design and
    # a static number reads as "nothing is happening". Report the period it
    # actually covers, which does move.
    ds_file = d / "dataset.jsonl"
    if ds_file.exists():
        first_ts = last_ts = None
        try:
            with ds_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ts = int(json.loads(line).get("ts", 0) or 0)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                    if ts <= 0:
                        continue
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts
        except OSError:
            first_ts = last_ts = None
        if first_ts and last_ts and last_ts > first_ts:
            info["ts_from"] = first_ts
            info["ts_to"] = last_ts
            info["span_hours"] = (last_ts - first_ts) / 3_600_000
    return info


def candle_stats(raw_dir: str, symbols: list[str]) -> list[dict[str, Any]]:
    """Per-symbol normalized-candle counts, synthetic ratio and last update age."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    out: list[dict[str, Any]] = []
    for symbol in symbols:
        total = 0
        synthetic = 0
        last_ts = 0
        for part in sorted((Path(raw_dir) / "normalized" / symbol).glob("part-*.jsonl")):
            with part.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    total += 1
                    if row.get("is_synthetic"):
                        synthetic += 1
                    ts = int(row.get("ts", 0) or 0)
                    if ts > last_ts:
                        last_ts = ts
        age_min = None if last_ts == 0 else max(0.0, (now_ms - last_ts) / 60_000)
        out.append(
            {
                "symbol": symbol,
                "candles": total,
                "synthetic_pct": (100.0 * synthetic / total) if total else 0.0,
                "last_update_min": age_min,
            }
        )
    return out


def load_candles(raw_dir: str, symbol: str, limit: int = 500) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for part in sorted((Path(raw_dir) / "normalized" / symbol).glob("part-*.jsonl")):
        rows.extend(_read_jsonl_tail(part, max_lines=limit * 4))
    rows.sort(key=lambda r: int(r.get("ts", 0) or 0))
    return rows[-limit:]


def drift_report(features_dir: str, model: dict[str, Any] | None, symbols: list[str]) -> list[dict[str, Any]]:
    from amber.monitoring.drift import detect_drift

    reference = model.get("train_reference") if isinstance(model, dict) else None
    if not isinstance(reference, dict) or not reference:
        reference = None
    out: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            res = detect_drift(Path(features_dir), symbol, reference=reference)
        except Exception as exc:
            res = {"drift": False, "level": "error", "max_psi": None, "per_feature": {}, "error": str(exc)}
        drifting = sorted(
            (name for name, v in res.get("per_feature", {}).items() if v > 0.1),
            key=lambda n: res["per_feature"][n],
            reverse=True,
        )
        out.append(
            {
                "symbol": symbol,
                "level": res.get("level", "low"),
                "max_psi": res.get("max_psi"),
                "drifting_features": ", ".join(drifting) if drifting else "—",
                "reference": res.get("reference", "n/a"),
            }
        )
    return out


def signal_direction(sig: dict[str, Any]) -> str:
    up = float(sig.get("prob_up_calibrated", 0.0) or 0.0)
    dn = float(sig.get("prob_down_calibrated", 0.0) or 0.0)
    return "pump" if up >= dn else "dump"


def signal_top_drivers(sig: dict[str, Any], n: int = 3) -> str:
    impacts = sig.get("explanation", {}).get("top_feature_impacts", []) if isinstance(sig.get("explanation"), dict) else []
    parts: list[str] = []
    for item in impacts[:n]:
        if isinstance(item, dict) and item:
            name, value = next(iter(item.items()))
            parts.append(f"{name}={value:+.3f}")
    return ", ".join(parts) if parts else "—"


def load_threshold_sweep(logs_dir: str) -> dict[str, Any] | None:
    """Latest operating-threshold sweep, computed by the pipeline on a schedule."""
    try:
        from amber.backtest.tuning import load_sweep

        return load_sweep(Path(logs_dir))
    except Exception:
        return None
