"""Shadow scanner: the one-feature rule the model has to beat, run live.

The baseline check found the move model indistinguishable from ranking rows by
raw `range_atr_14`. That was measured on a few hours of test segment. The
forward ledger keeps asking, on live data, by running the rule next to the
model under identical conditions:

- the same universe, warm-up, synthetic-bar and spread filters;
- the same per-symbol cooldown and concurrency cap (a separate gate, so the
  two never take each other's slots);
- the same alert RATE. At each retrain the rule's cut is set so that it would
  fire on the same fraction of test-segment rows as the model's operating
  threshold did. The cut is fixed until the next retrain, so the rule chooses
  its own moments — it does not inherit the model's timing.

Shadow alerts are logged, never routed to Telegram or Discord.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RULE_FEATURE = "range_atr_14"
RULE_FILE = "shadow_rule.json"
SHADOW_SIGNALS_FILE = "shadow_signals.jsonl"
GATE_STATE_KEY = "shadow_signal_gate"


def fit_shadow_rule(
    oos_rows: list[dict[str, Any]],
    alert_fraction: float,
    *,
    model_run_id: str | None,
    feature: str = RULE_FEATURE,
) -> dict[str, Any]:
    """The feature cut that fires on `alert_fraction` of the test segment."""
    values = sorted(
        (float(r[feature]) for r in oos_rows if r.get(feature) is not None),
        reverse=True,
    )
    take = int(round(len(values) * alert_fraction))
    rule: dict[str, Any] = {
        "feature": feature,
        "alert_fraction": alert_fraction,
        "rows": len(values),
        "model_run_id": model_run_id,
    }
    if take <= 0 or not values:
        # The model fires on nothing at its threshold; so does the rule.
        rule["threshold"] = None
        return rule
    rule["threshold"] = values[min(take, len(values)) - 1]
    return rule


def save_shadow_rule(logs_root: Path, rule: dict[str, Any]) -> None:
    logs_root.mkdir(parents=True, exist_ok=True)
    tmp = logs_root / (RULE_FILE + ".tmp")
    tmp.write_text(json.dumps(rule), encoding="utf-8")
    tmp.replace(logs_root / RULE_FILE)


def load_shadow_rule(logs_root: Path) -> dict[str, Any] | None:
    try:
        rule = json.loads((logs_root / RULE_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return rule if isinstance(rule, dict) and rule.get("threshold") is not None else None


def _spread_bps(row: dict[str, Any]) -> float:
    """Same arithmetic as filters.spread_bps, applied to the feature row the
    signal's market_context is copied from."""
    bid = float(row.get("bid", 0.0) or 0.0)
    ask = float(row.get("ask", 0.0) or 0.0)
    mid = float(row.get("mid_price", 0.0) or 0.0)
    if bid <= 0 or ask <= 0 or mid <= 0:
        return float(row.get("spread_bps", 0.0) or 0.0)
    return ((ask - bid) / mid) * 10_000


def shadow_candidates(
    feature_rows: list[dict[str, Any]],
    rule: dict[str, Any],
    *,
    min_warmup: int,
    spread_max_bps: float,
) -> list[dict[str, Any]]:
    """Rows over the rule's cut, strongest first (as the scanner ranks)."""
    feature, cut = rule["feature"], float(rule["threshold"])
    out = []
    for row in feature_rows:
        if bool(row.get("is_synthetic", False)) or int(row.get("obs", 0) or 0) < min_warmup:
            continue
        value = row.get(feature)
        if value is None or float(value) < cut or _spread_bps(row) > spread_max_bps:
            continue
        out.append(row)
    out.sort(key=lambda r: float(r[feature]), reverse=True)
    return out


def append_shadow_signal(
    logs_root: Path,
    row: dict[str, Any],
    rule: dict[str, Any],
    *,
    horizon_min: int,
    target_pct: float,
) -> None:
    rec = {
        "event_ts": int(row["ts"]),
        "symbol": row["symbol"],
        "horizon_min": horizon_min,
        "target_up_pct": target_pct,
        rule["feature"]: float(row[rule["feature"]]),
        "threshold": rule["threshold"],
        "model_run_id": rule.get("model_run_id"),
    }
    logs_root.mkdir(parents=True, exist_ok=True)
    with (logs_root / SHADOW_SIGNALS_FILE).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
