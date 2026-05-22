from __future__ import annotations

import json
<<<<<<< HEAD
import math
=======
>>>>>>> origin/main
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
<<<<<<< HEAD
        for lineno, line in enumerate(fh, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {lineno}: {exc.msg}") from exc
=======
        for line in fh:
            rows.append(json.loads(line))
>>>>>>> origin/main
    return rows


def event_backtest(datasets_root: Path, slippage_bps: float = 5.0, fee_bps: float = 4.0) -> dict[str, Any]:
<<<<<<< HEAD
    if not math.isfinite(slippage_bps) or not math.isfinite(fee_bps):
        raise ValueError("slippage_bps and fee_bps must be finite numbers")
    if slippage_bps < 0 or fee_bps < 0:
        raise ValueError("slippage_bps and fee_bps must be non-negative")

    candidates = sorted([p for p in datasets_root.iterdir() if p.is_dir() and p.name.startswith("dataset_")])
    if not candidates:
        raise ValueError(f"No dataset_* directories found under: {datasets_root}")
    latest = candidates[-1]
    dataset_file = latest / "dataset.jsonl"
    if not dataset_file.exists():
        raise ValueError(f"Missing dataset file: {dataset_file}")
    rows = _read_jsonl(dataset_file)
=======
    latest = sorted([p for p in datasets_root.iterdir() if p.is_dir() and p.name.startswith("dataset_")])[-1]
    rows = _read_jsonl(latest / "dataset.jsonl")
>>>>>>> origin/main
    if not rows:
        raise ValueError("Dataset empty")

    cost = (slippage_bps + fee_bps) / 10_000
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

    precision = counts["TP"] / len(rows)
<<<<<<< HEAD
    resolution_rate = (counts["TP"] + counts["SL"]) / len(rows)
    win_rate = counts["TP"] / (counts["TP"] + counts["SL"]) if (counts["TP"] + counts["SL"]) else 0.0
    avg_pnl = mean(pnls)
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
    expectancy = avg_pnl
=======
    avg_pnl = mean(pnls)
    vol = pstdev(pnls) if len(pnls) > 1 else 0.0
    sharpe = 0.0 if vol == 0 else avg_pnl / vol
>>>>>>> origin/main

    # simple cumulative max drawdown
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    return {
        "dataset_run": latest.name,
        "signals": len(rows),
        "precision": precision,
<<<<<<< HEAD
        "resolution_rate": resolution_rate,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "sharpe": sharpe,
        "profit_factor": profit_factor,
        "payoff_ratio": payoff_ratio,
        "expectancy": expectancy,
=======
        "avg_pnl": avg_pnl,
        "sharpe": sharpe,
>>>>>>> origin/main
        "max_drawdown": max_dd,
        "tp": counts["TP"],
        "sl": counts["SL"],
        "timeout": counts["Timeout"],
    }
