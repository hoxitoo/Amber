from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def event_backtest(datasets_root: Path, slippage_bps: float = 5.0, fee_bps: float = 4.0) -> dict[str, Any]:
    latest = sorted([p for p in datasets_root.iterdir() if p.is_dir() and p.name.startswith("dataset_")])[-1]
    rows = _read_jsonl(latest / "dataset.jsonl")
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
    avg_pnl = mean(pnls)
    vol = pstdev(pnls) if len(pnls) > 1 else 0.0
    sharpe = 0.0 if vol == 0 else avg_pnl / vol

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
        "avg_pnl": avg_pnl,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "tp": counts["TP"],
        "sl": counts["SL"],
        "timeout": counts["Timeout"],
    }
