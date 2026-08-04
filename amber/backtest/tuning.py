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

from amber.models.dataset_io import load_latest_dataset_rows, order_with_pseudo_time, split_rows
from amber.models.eval import _load_latest_calibration
from amber.models.infer import infer_row_prob, load_latest_model
from amber.signals.filters import base_rate_for, effective_prob_min
from amber.signals.scorer import calibrated_prob_for_target, coherent_pump_dump

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
    cost: float,
) -> dict[str, float]:
    """Book trades the way the backtester does: 1-bar entry lag, one open
    position per symbol, both barriers, cost charged on every trade."""
    pnl: list[float] = []
    tp = sl = timeout = 0
    open_until: dict[str, int] = {}
    pending: dict[str, str] = {}

    for r, (up, dn) in zip(rows, probs):
        sym = str(r.get("symbol", ""))
        if sym in pending:
            side = pending.pop(sym)
            hit = int(r.get("up_hit", 0)) if side == "pump" else int(r.get("down_hit", 0))
            miss = int(r.get("down_hit", 0)) if side == "pump" else int(r.get("up_hit", 0))
            target = float(r.get("up_pct", 0.002))
            if hit:
                pnl.append(target - cost)
                tp += 1
            elif miss:
                pnl.append(-target - cost)
                sl += 1
            else:
                pnl.append(-cost)
                timeout += 1
            open_until[sym] = int(r.get("horizon_steps", 0) or 0)
            continue
        if open_until.get(sym, 0) > 0:
            open_until[sym] -= 1
            continue
        if up >= up_min and (up - dn) >= dir_min:
            pending[sym] = "pump"
        elif dn >= dn_min and (dn - up) >= dir_min:
            pending[sym] = "dump"

    n = len(pnl)
    if n == 0:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "resolved": 0.0}
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

    def score(seg_rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
        out = []
        for r in seg_rows:
            up = calibrated_prob_for_target(infer_row_prob(model, r, target="pump"), calib, target="pump")
            dn = calibrated_prob_for_target(infer_row_prob(model, r, target="dump"), calib, target="dump")
            out.append(coherent_pump_dump(up, dn))
        return out

    p_select, p_verify = score(select), score(verify)

    grid: list[dict[str, Any]] = []
    for lift in lifts:
        thr = {"prob_lift_min": lift, "prob_abs_floor": 0.0}
        up_min = effective_prob_min(thr, base_up, absolute_key="pump_prob_calibrated_min")
        dn_min = effective_prob_min(thr, base_dn, absolute_key="dump_prob_calibrated_min")
        for dir_min in dir_mins:
            grid.append({
                "prob_lift_min": lift,
                "directional_score_min": dir_min,
                "up_min": up_min,
                "down_min": dn_min,
                "selection": _replay(select, p_select, up_min, dn_min, dir_min, cost),
                "validation": _replay(verify, p_verify, up_min, dn_min, dir_min, cost),
            })

    eligible = [g for g in grid if g["selection"]["trades"] >= MIN_TRADES]
    best = max(eligible, key=lambda g: g["selection"]["expectancy"]) if eligible else None

    result: dict[str, Any] = {
        "status": "ok",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "dataset_run": dataset_run,
        "horizon_steps": horizon,
        "cost_bps": slippage_bps + fee_bps,
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
