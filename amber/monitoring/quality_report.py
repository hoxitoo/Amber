from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from amber.monitoring.drift import PredictionBiasMonitor, RollingAUCMonitor, psi_monitor


def _read_signals_jsonl(path: Path) -> list[dict[str, Any]]:
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
def build_quality_report(signals_path: Path) -> dict[str, Any]:
    auc_m = RollingAUCMonitor(window=200)
    bias_m = PredictionBiasMonitor(window=200)

    rows = _read_signals_jsonl(signals_path)

    valid_up: list[float] = []
    for r in rows:
        p_up = _safe_prob(r.get("prob_up_calibrated", 0.0))
        p_dn = _safe_prob(r.get("prob_down_calibrated", 0.0))
        if p_up is None or p_dn is None:
            continue
        valid_up.append(p_up)
        bias_m.update(p_up, p_dn)
        # placeholder pseudo outcome
        auc_m.update(int(p_up > p_dn), p_up)

    up = valid_up
    mid = len(up) // 2
    psi_result = psi_monitor(up[:mid], up[mid:]) if len(up) >= 20 else {"psi": 0.0, "level": "low"}

    return {
        "signals": len(rows),
        "rolling_auc": auc_m.value(),
        "prediction_bias": bias_m.bias(),
        "psi": psi_result,
    }
