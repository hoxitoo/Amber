"""Operating-threshold sweep with out-of-sample validation.

Selecting the best-looking threshold on the same data you then report is how a
backtest flatters itself. The sweep therefore *selects* on the calibration
segment and *validates* on the test segment, and reports both. A point that wins
on selection but fails validation was fitted to noise, and is labelled as such
rather than presented as an edge.

Deliberately does not apply anything: re-running a search every day and adopting
whatever currently validates is multiple testing by another name, so adoption
stays a human decision.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from amber.backtest.backtester import operating_point, replay_with_probs, score_rows
from amber.models.dataset_io import load_latest_dataset_rows, order_with_pseudo_time, split_rows
from amber.models.eval import _load_latest_calibration
from amber.models.infer import load_latest_model
from amber.signals.filters import base_rate_for

SWEEP_FILE = "threshold_sweep.json"
DEFAULT_LIFTS = (1.2, 1.5, 2.0, 2.5, 3.0)
DEFAULT_DIRS = (0.0, 0.05, 0.10)
MIN_TRADES = 20


def _replay(
    rows: list[dict[str, Any]],
    probs: list[tuple[float, float]],
    up_min: float,
    dn_min: float,
    dir_min: float,
    spread_max: float,
    cost: float,
) -> dict[str, float]:
    """Score one grid point through the *same* trading rules the backtest uses.

    This delegates rather than reimplementing: the sweep used to keep its own
    copy, which had silently lost the spread filter, so it was validating a gate
    the live scanner does not run and could bless a threshold that behaves
    differently in production.
    """
    pnl, counts, _extra = replay_with_probs(
        rows, probs, up_min=up_min, down_min=dn_min, dir_min=dir_min, spread_max=spread_max, cost=cost
    )
    n = len(pnl)
    if n == 0:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "resolved": 0.0}
    tp, sl = counts["TP"], counts["SL"]
    gross_profit = sum(p for p in pnl if p > 0)
    gross_loss = abs(sum(p for p in pnl if p < 0))
    return {
        "trades": n,
        "win_rate": tp / (tp + sl) if (tp + sl) else 0.0,
        "profit_factor": (gross_profit / gross_loss) if gross_loss else float("inf"),
        "expectancy": sum(pnl) / n,
        "resolved": (tp + sl) / n,
    }


def sweep_thresholds(
    models_root: Path,
    datasets_root: Path,
    *,
    slippage_bps: float = 5.0,
    fee_bps: float = 4.0,
    lifts: tuple[float, ...] = DEFAULT_LIFTS,
    dir_mins: tuple[float, ...] = DEFAULT_DIRS,
    live_thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sweep the grid, returning every point plus a verdict on the best one."""
    model = load_latest_model(models_root)
    calib = _load_latest_calibration(models_root)
    all_rows, dataset_run = load_latest_dataset_rows(datasets_root)
    rows, pseudo_ts, _ = order_with_pseudo_time(all_rows)

    splits = model.get("splits")
    if not isinstance(splits, dict):
        return {"status": "no_splits", "reason": "model has no holdout splits; cannot validate honestly"}

    seg = split_rows(rows, pseudo_ts, splits)
    horizons = sorted({int(r.get("horizon_steps", 0) or 0) for r in rows})
    horizon = horizons[0] if horizons else 0
    select = [r for r in seg["calib"] if int(r.get("horizon_steps", 0) or 0) == horizon]
    verify = [r for r in seg["test"] if int(r.get("horizon_steps", 0) or 0) == horizon]
    if not select or not verify:
        return {"status": "not_enough_data", "reason": "calibration or test segment is empty"}

    cost = (slippage_bps + fee_bps) / 10_000
    base_up = base_rate_for(model, "pump")
    base_dn = base_rate_for(model, "dump")

    p_select = score_rows(select, model, calib)
    p_verify = score_rows(verify, model, calib)
    # The sweep must model the gate that actually runs, spread filter included.
    spread_max = operating_point(model, live_thresholds or {})["spread_max"]

    grid: list[dict[str, Any]] = []
    for lift in lifts:
        op = operating_point(model, {**(live_thresholds or {}), "prob_lift_min": lift})
        for dir_min in dir_mins:
            grid.append({
                "prob_lift_min": lift,
                "directional_score_min": dir_min,
                "up_min": op["up_min"],
                "down_min": op["down_min"],
                "selection": _replay(select, p_select, op["up_min"], op["down_min"], dir_min, spread_max, cost),
                "validation": _replay(verify, p_verify, op["up_min"], op["down_min"], dir_min, spread_max, cost),
            })

    eligible = [g for g in grid if g["selection"]["trades"] >= MIN_TRADES]
    best = max(eligible, key=lambda g: g["selection"]["expectancy"]) if eligible else None

    result: dict[str, Any] = {
        "status": "ok",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "dataset_run": dataset_run,
        "horizon_steps": horizon,
        "cost_bps": slippage_bps + fee_bps,
        "spread_bps_max": spread_max,
        "base_rate_up": base_up,
        "base_rate_down": base_dn,
        "selection_rows": len(select),
        "validation_rows": len(verify),
        "grid": grid,
        "best": best,
    }
    if best is None:
        result["verdict"] = "no_candidate"
        result["verdict_text"] = f"Ни одна точка не дала {MIN_TRADES}+ сделок на отборочном сегменте."
    elif best["validation"]["expectancy"] > 0 and best["validation"]["profit_factor"] > 1.0:
        result["verdict"] = "holds"
        result["verdict_text"] = "Точка подтвердилась на невиданных данных — можно применять."
    else:
        result["verdict"] = "does_not_hold"
        result["verdict_text"] = (
            "Лучшая точка НЕ подтвердилась out-of-sample: результат отбора был шумом. "
            "Применять её — подгонка под бэктест, а не поиск края."
        )
    return result


def save_sweep(logs_dir: Path, result: dict[str, Any]) -> None:
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / SWEEP_FILE
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def load_sweep(logs_dir: Path) -> dict[str, Any] | None:
    path = Path(logs_dir) / SWEEP_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None
