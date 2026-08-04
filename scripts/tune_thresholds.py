"""Print the operating-threshold sweep.

The pipeline runs this automatically (see `pipeline.tune_min`) and the dashboard
shows the result, so this script is only for inspecting the full grid by hand.

    python scripts/tune_thresholds.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.backtest.tuning import save_sweep, sweep_thresholds
from amber.common.config import ConfigLoader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--fee-bps", type=float, default=4.0)
    args = ap.parse_args()

    cfg = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    storage = cfg["storage"]
    res = sweep_thresholds(
        Path(storage["models_dir"]),
        Path(storage["datasets_dir"]),
        slippage_bps=args.slippage_bps,
        fee_bps=args.fee_bps,
    )
    if res.get("status") != "ok":
        print(f"Развёртка недоступна: {res.get('reason', res.get('status'))}")
        return
    save_sweep(Path(storage["logs_dir"]), res)

    print(f"датасет {res['dataset_run']} · горизонт {res['horizon_steps']} свечей · издержки {res['cost_bps']:.0f} bps")
    print(f"отбор (calib): {res['selection_rows']:,} строк · проверка (test): {res['validation_rows']:,} строк")
    print(f"базовые частоты: pump {res['base_rate_up']:.3f} · dump {res['base_rate_down']:.3f}\n")
    print(f"{'lift':>5} {'dir':>5} | {'ОТБОР (calib)':^32} | {'ПРОВЕРКА (test)':^32}")
    print(f"{'':>5} {'':>5} | {'сделок':>7} {'win':>6} {'PF':>6} {'exp bps':>8} | {'сделок':>7} {'win':>6} {'PF':>6} {'exp bps':>8}")
    print("-" * 88)
    for g in res["grid"]:
        s, v = g["selection"], g["validation"]
        print(
            f"{g['prob_lift_min']:>5.1f} {g['directional_score_min']:>5.2f} | "
            f"{s['trades']:>7,} {s['win_rate']:>6.3f} {s['profit_factor']:>6.2f} {s['expectancy']*1e4:>8.2f} | "
            f"{v['trades']:>7,} {v['win_rate']:>6.3f} {v['profit_factor']:>6.2f} {v['expectancy']*1e4:>8.2f}"
        )

    best = res.get("best")
    if best:
        v = best["validation"]
        print(f"\nЛучшая на ОТБОРЕ: prob_lift_min={best['prob_lift_min']}, "
              f"directional_score_min={best['directional_score_min']}")
        print(f"  на ПРОВЕРКЕ: {v['trades']:,} сделок, PF {v['profit_factor']:.2f}, "
              f"матожидание {v['expectancy']*1e4:+.2f} bps")
    print(f"\n{res['verdict_text']}")


if __name__ == "__main__":
    main()
