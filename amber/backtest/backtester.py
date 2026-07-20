from __future__ import annotations

import logging
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from amber.models.dataset_io import load_latest_dataset_rows, order_with_pseudo_time, split_rows

logger = logging.getLogger(__name__)


def _aggregate(pnls: list[float], counts: dict[str, int], extra: dict[str, Any]) -> dict[str, Any]:
    n = len(pnls)
    precision = counts["TP"] / n if n else 0.0
    resolution_rate = (counts["TP"] + counts["SL"]) / n if n else 0.0
    win_rate = counts["TP"] / (counts["TP"] + counts["SL"]) if (counts["TP"] + counts["SL"]) else 0.0
    avg_pnl = mean(pnls) if pnls else 0.0
    vol = pstdev(pnls) if len(pnls) > 1 else 0.0
    sharpe = 0.0 if vol == 0 else avg_pnl / vol
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = float("inf") if gross_loss == 0 and gross_profit > 0 else 0.0 if gross_loss == 0 else gross_profit / gross_loss
    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [abs(p) for p in pnls if p < 0]
    avg_win = mean(win_pnls) if win_pnls else 0.0
    avg_loss = mean(loss_pnls) if loss_pnls else 0.0
    payoff_ratio = float("inf") if avg_loss == 0 and avg_win > 0 else 0.0 if avg_loss == 0 else avg_win / avg_loss

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    return {
        "signals": n,
        "precision": precision,
        "resolution_rate": resolution_rate,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "sharpe": sharpe,
        "profit_factor": profit_factor,
        "payoff_ratio": payoff_ratio,
        "expectancy": avg_pnl,
        "max_drawdown": max_dd,
        "tp": counts["TP"],
        "sl": counts["SL"],
        "timeout": counts["Timeout"],
        **extra,
    }


def _label_replay(rows: list[dict[str, Any]], cost: float) -> tuple[list[float], dict[str, int]]:
    """Legacy mode: book the labeled outcome of every event row (base-rate view,
    no model involved)."""
    pnls: list[float] = []
    counts = {"TP": 0, "SL": 0, "Timeout": 0}
    for r in rows:
        up = int(r.get("up_hit", 0))
        down = int(r.get("down_hit", 0))
        target = float(r.get("up_pct", 0.002))
        if up == 1 and down == 0:
            pnl = target - cost
            counts["TP"] += 1
        elif down == 1 and up == 0:
            pnl = -target - cost
            counts["SL"] += 1
        else:
            pnl = -cost
            counts["Timeout"] += 1
        pnls.append(pnl)
    return pnls, counts


def _signal_replay(
    rows: list[dict[str, Any]],
    model: dict[str, Any],
    calibration: dict[str, Any],
    thresholds: dict[str, Any],
    cost: float,
) -> tuple[list[float], dict[str, int], dict[str, Any]]:
    """Model-driven mode: trade only where calibrated probabilities pass the
    real threshold chain; one open trade per symbol at a time."""
    from amber.models.infer import infer_row_prob
    from amber.signals.scorer import calibrated_prob_for_target

    up_min = float(thresholds.get("pump_prob_calibrated_min", 0.65))
    down_min = float(thresholds.get("dump_prob_calibrated_min", 0.65))
    dir_min = float(thresholds.get("directional_score_min", 0.2))
    spread_max = float(thresholds.get("spread_bps_max", 30.0))

    pnls: list[float] = []
    counts = {"TP": 0, "SL": 0, "Timeout": 0}
    directions = {"pump": 0, "dump": 0}
    open_until: dict[str, int] = {}  # symbol -> countdown of remaining candles
    evaluated = 0

    for r in rows:
        symbol = str(r.get("symbol", ""))
        if open_until.get(symbol, 0) > 0:
            open_until[symbol] -= 1
            continue
        evaluated += 1

        if float(r.get("spread_bps", 0.0) or 0.0) > spread_max:
            continue

        up_cal = calibrated_prob_for_target(infer_row_prob(model, r, target="pump"), calibration, target="pump")
        down_cal = calibrated_prob_for_target(infer_row_prob(model, r, target="dump"), calibration, target="dump")

        direction: str | None = None
        if up_cal >= up_min and (up_cal - down_cal) >= dir_min:
            direction = "pump"
        elif down_cal >= down_min and (down_cal - up_cal) >= dir_min:
            direction = "dump"
        if direction is None:
            continue

        up = int(r.get("up_hit", 0))
        down = int(r.get("down_hit", 0))
        target_up = float(r.get("up_pct", 0.002))
        target_down = float(r.get("down_pct", 0.002))

        if direction == "pump":
            if up == 1:
                pnl = target_up - cost
                counts["TP"] += 1
            elif down == 1:
                pnl = -target_down - cost
                counts["SL"] += 1
            else:
                pnl = -cost
                counts["Timeout"] += 1
        else:
            if down == 1:
                pnl = target_down - cost
                counts["TP"] += 1
            elif up == 1:
                pnl = -target_up - cost
                counts["SL"] += 1
            else:
                pnl = -cost
                counts["Timeout"] += 1

        directions[direction] += 1
        pnls.append(pnl)
        open_until[symbol] = int(r.get("horizon_steps", 0) or 0)

    extra = {
        "mode": "model_signals",
        "rows_evaluated": evaluated,
        "trades_pump": directions["pump"],
        "trades_dump": directions["dump"],
    }
    return pnls, counts, extra


def event_backtest(
    datasets_root: Path,
    models_root: Path | None = None,
    slippage_bps: float = 5.0,
    fee_bps: float = 4.0,
    *,
    thresholds: dict[str, Any] | None = None,
    horizon: int | None = None,
) -> dict[str, Any]:
    """Event backtest over the latest dataset.

    With `models_root` given, replays the trained model + calibration through the
    real threshold chain on the test segment (promotion-gate mode). Without it,
    falls back to the legacy label replay, which measures event base rates only.
    """
    if not math.isfinite(slippage_bps) or not math.isfinite(fee_bps):
        raise ValueError("slippage_bps and fee_bps must be finite numbers")
    if slippage_bps < 0 or fee_bps < 0:
        raise ValueError("slippage_bps and fee_bps must be non-negative")

    all_rows, dataset_run = load_latest_dataset_rows(datasets_root)
    if not all_rows:
        raise ValueError("Dataset empty")
    cost = (slippage_bps + fee_bps) / 10_000

    if models_root is None:
        pnls, counts = _label_replay(all_rows, cost)
        return {"dataset_run": dataset_run, "mode": "label_replay", **_aggregate(pnls, counts, {})}

    from amber.models.eval import _load_latest_calibration
    from amber.models.infer import load_latest_model

    model = load_latest_model(models_root)
    calibration = _load_latest_calibration(models_root)

    rows, pseudo_ts, _mode = order_with_pseudo_time(all_rows)
    splits = model.get("splits")
    segment = "full_in_sample"
    if isinstance(splits, dict):
        test_rows = split_rows(rows, pseudo_ts, splits)["test"]
        if test_rows:
            rows = test_rows
            segment = "test_split"
        else:
            logger.warning("model splits produced empty test segment; backtesting in-sample")
    else:
        logger.warning("model has no holdout splits; backtest runs in-sample")

    # One horizon per trade: overlapping multi-horizon rows would book the same
    # underlying move several times.
    horizons = sorted({int(r.get("horizon_steps", 0) or 0) for r in rows})
    chosen = horizon if horizon is not None else (horizons[0] if horizons else 0)
    rows = [r for r in rows if int(r.get("horizon_steps", 0) or 0) == chosen]
    if not rows:
        raise ValueError(f"No dataset rows for horizon={chosen}")

    pnls, counts, extra = _signal_replay(rows, model, calibration, thresholds or {}, cost)
    extra.update({"segment": segment, "horizon_steps": chosen})
    return {"dataset_run": dataset_run, **_aggregate(pnls, counts, extra)}
