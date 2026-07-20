from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
from typing import Any

from amber.monitoring.drift import PredictionBiasMonitor, RollingAUCMonitor, psi_from_quantile_reference

logger = logging.getLogger(__name__)


def _read_jsonl_tolerant(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _safe_prob(value: Any) -> float | None:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p):
        return None
    return max(0.0, min(1.0, p))


def _event_ts_ms(value: Any) -> int | None:
    """Signal `event_ts` arrives as an ISO datetime string (SignalV1 JSON) or an
    epoch-ms int."""
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return int(value)
    if isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            return None
    return None


class _CandleIndex:
    """Lazy per-symbol index of normalized candles for outcome confirmation."""

    def __init__(self, raw_root: Path) -> None:
        self.raw_root = raw_root
        self._cache: dict[str, list[dict[str, Any]]] = {}

    def candles(self, symbol: str) -> list[dict[str, Any]]:
        if symbol not in self._cache:
            rows: list[dict[str, Any]] = []
            for part in sorted((self.raw_root / "normalized" / symbol).glob("part-*.jsonl")):
                rows.extend(_read_jsonl_tolerant(part))
            rows.sort(key=lambda r: int(r.get("ts", 0) or 0))
            self._cache[symbol] = rows
        return self._cache[symbol]


def _confirmed_outcome(
    index: _CandleIndex,
    symbol: str,
    event_ts: int,
    horizon_candles: int,
    target_up_pct: float,
    step_ms: int = 60_000,
) -> int | None:
    """1/0 if the pump target was/wasn't hit within the horizon; None while the
    horizon has not fully elapsed in the data (unconfirmed)."""
    candles = index.candles(symbol)
    if not candles:
        return None
    ts_list = [int(c.get("ts", 0) or 0) for c in candles]
    entry_i = bisect_right(ts_list, event_ts) - 1
    if entry_i < 0:
        return None
    entry_price = float(candles[entry_i].get("close", 0.0) or 0.0)
    if entry_price <= 0 or target_up_pct <= 0:
        return None
    deadline = event_ts + horizon_candles * step_ms
    if ts_list[-1] < deadline:
        return None  # horizon not yet elapsed -> outcome unknown
    target = entry_price * (1.0 + target_up_pct)
    for c in candles[entry_i + 1 :]:
        ts = int(c.get("ts", 0) or 0)
        if ts > deadline:
            break
        if float(c.get("high", 0.0) or 0.0) >= target:
            return 1
    return 0


def _feature_psi(
    models_root: Path | None,
    features_root: Path | None,
    window: int = 500,
) -> dict[str, Any]:
    if models_root is None or features_root is None:
        return {"level": "unavailable", "max_psi": None, "per_feature": {}, "reason": "no_reference_configured"}
    try:
        from amber.models.infer import load_latest_model

        reference = load_latest_model(models_root).get("train_reference")
    except Exception:
        reference = None
    if not isinstance(reference, dict) or not reference:
        return {"level": "unavailable", "max_psi": None, "per_feature": {}, "reason": "model_has_no_train_reference"}

    live_by_feature: dict[str, list[float]] = {name: [] for name in reference}
    features_dir = features_root / "features"
    if features_dir.exists():
        for part in sorted(features_dir.glob("*/part-*.jsonl")):
            for row in _read_jsonl_tolerant(part)[-window:]:
                for name in reference:
                    live_by_feature[name].append(float(row.get(name, 0.0) or 0.0))

    per_feature: dict[str, float] = {}
    for name, edges in reference.items():
        live = live_by_feature.get(name, [])[-window:]
        if len(live) >= 20:
            per_feature[name] = psi_from_quantile_reference(list(edges), live)

    if not per_feature:
        return {"level": "unavailable", "max_psi": None, "per_feature": {}, "reason": "not_enough_live_rows"}
    max_psi = max(per_feature.values())
    level = "high" if max_psi > 0.2 else "medium" if max_psi > 0.1 else "low"
    return {"level": level, "max_psi": max_psi, "per_feature": per_feature, "reason": "ok"}


def build_quality_report(
    signals_path: Path,
    raw_root: Path | None = None,
    models_root: Path | None = None,
    features_root: Path | None = None,
) -> dict[str, Any]:
    """Model-quality snapshot from emitted signals.

    - `rolling_auc` is computed only against confirmed real outcomes (signal
      joined with normalized candles after its horizon elapsed); without
      `raw_root` it is None rather than a fabricated number.
    - `psi` compares live feature distributions against the train reference
      stored in the model artifact.
    """
    auc_m = RollingAUCMonitor(window=200)
    bias_m = PredictionBiasMonitor(window=200)

    rows = _read_jsonl_tolerant(signals_path)
    index = _CandleIndex(raw_root) if raw_root is not None else None

    confirmed = 0
    unconfirmed = 0
    for r in rows:
        p_up = _safe_prob(r.get("prob_up_calibrated", 0.0))
        p_dn = _safe_prob(r.get("prob_down_calibrated", 0.0))
        if p_up is None or p_dn is None:
            continue
        bias_m.update(p_up, p_dn)

        if index is None:
            continue
        event_ts = _event_ts_ms(r.get("event_ts"))
        symbol = str(r.get("symbol", ""))
        horizon = int(r.get("horizon_min", 0) or 0)
        target_up = float(r.get("target_up_pct", 0.0) or 0.0)
        if event_ts is None or not symbol or horizon <= 0:
            continue
        outcome = _confirmed_outcome(index, symbol, event_ts, horizon, target_up)
        if outcome is None:
            unconfirmed += 1
            continue
        confirmed += 1
        auc_m.update(outcome, p_up)

    return {
        "signals": len(rows),
        "rolling_auc": auc_m.value(),
        "auc_confirmed_outcomes": confirmed,
        "auc_unconfirmed_outcomes": unconfirmed,
        "prediction_bias": bias_m.bias(),
        "psi": _feature_psi(models_root, features_root),
    }
