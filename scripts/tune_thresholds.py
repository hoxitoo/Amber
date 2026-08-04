"""Sweep operating thresholds, then validate the choice on data it never saw.

Picking the best-looking point on the same segment you measure is how a backtest
flatters itself. This selects on the calibration segment and reports the chosen
point's performance on the test segment — if the two disagree, the "best"
threshold was fitted to noise, and the script says so.

    python scripts/tune_thresholds.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
from amber.models.dataset_io import load_latest_dataset_rows, order_with_pseudo_time, split_rows
from amber.models.eval import _load_latest_calibration
from amber.models.infer import infer_row_prob, load_latest_model
from amber.signals.filters import base_rate_for
from amber.signals.scorer import calibrated_prob_for_target, coherent_pump_dump


def _replay(rows, probs, lift, dir_min, base_up, base_dn, cost):
    """Book trades exactly like the backtester: 1-bar entry lag, one open
    position per symbol, both barriers, cost on every trade."""
    up_min, dn_min = lift * base_up, lift * base_dn
    pnl, tp, sl, timeout = [], 0, 0, 0
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
    gp = sum(p for p in pnl if p > 0)
    gl = abs(sum(p for p in pnl if p < 0))
    return {
        "trades": n,
        "win_rate": tp / (tp + sl) if (tp + sl) else 0.0,
        "profit_factor": (gp / gl) if gl else float("inf"),
        "expectancy": sum(pnl) / n,
        "resolved": (tp + sl) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--fee-bps", type=float, default=4.0)
    args = ap.parse_args()

    root = Path.cwd()
    cfg = ConfigLoader(root).load_yaml("config/amber.yaml")
    models_root = Path(cfg["storage"]["models_dir"])
    datasets_root = Path(cfg["storage"]["datasets_dir"])
    cost = (args.slippage_bps + args.fee_bps) / 10_000

    model = load_latest_model(models_root)
    calib = _load_latest_calibration(models_root)
    all_rows, run = load_latest_dataset_rows(datasets_root)
    rows, pseudo_ts, _ = order_with_pseudo_time(all_rows)

    splits = model.get("splits")
    if not isinstance(splits, dict):
        print("Model has no holdout splits — cannot validate honestly. Train on more data first.")
        return
    seg = split_rows(rows, pseudo_ts, splits)
    horizon = sorted({int(r.get("horizon_steps", 0) or 0) for r in rows})[0]
    select = [r for r in seg["calib"] if int(r.get("horizon_steps", 0) or 0) == horizon]
    verify = [r for r in seg["test"] if int(r.get("horizon_steps", 0) or 0) == horizon]
    if not select or not verify:
        print("Empty calibration or test segment — not enough data yet.")
        return

    base_up = base_rate_for(model, "pump") or 0.1
    base_dn = base_rate_for(model, "dump") or 0.1

    def score(seg_rows):
        out = []
        for r in seg_rows:
            up = calibrated_prob_for_target(infer_row_prob(model, r, target="pump"), calib, target="pump")
            dn = calibrated_prob_for_target(infer_row_prob(model, r, target="dump"), calib, target="dump")
            out.append(coherent_pump_dump(up, dn))
        return out

    p_select, p_verify = score(select), score(verify)

    print(f"dataset {run} · horizon {horizon} bars · cost {cost*1e4:.0f} bps")
    print(f"selection segment (calib): {len(select):,} rows · validation (test): {len(verify):,} rows")
    print(f"base rates: pump {base_up:.3f} · dump {base_dn:.3f}\n")

    grid = [(lift, dm) for lift in (1.2, 1.5, 2.0, 2.5, 3.0) for dm in (0.0, 0.05, 0.10)]
    print(f"{'lift':>5} {'dir':>5} | {'SELECTION (calib)':^34} | {'VALIDATION (test)':^34}")
    print(f"{'':>5} {'':>5} | {'trades':>7} {'win':>6} {'PF':>6} {'exp bps':>9} | {'trades':>7} {'win':>6} {'PF':>6} {'exp bps':>9}")
    print("-" * 92)

    best = None
    for lift, dm in grid:
        s = _replay(select, p_select, lift, dm, base_up, base_dn, cost)
        v = _replay(verify, p_verify, lift, dm, base_up, base_dn, cost)
        print(f"{lift:>5.1f} {dm:>5.2f} | {s['trades']:>7,} {s['win_rate']:>6.3f} {s['profit_factor']:>6.2f} "
              f"{s['expectancy']*1e4:>9.2f} | {v['trades']:>7,} {v['win_rate']:>6.3f} {v['profit_factor']:>6.2f} "
              f"{v['expectancy']*1e4:>9.2f}")
        if s["trades"] >= 20 and (best is None or s["expectancy"] > best[0]["expectancy"]):
            best = (s, v, lift, dm)

    if best is None:
        print("\nNo threshold produced enough trades on the selection segment.")
        return
    s, v, lift, dm = best
    print(f"\nBest on SELECTION: prob_lift_min={lift}, directional_score_min={dm}")
    print(f"  selection : {s['trades']:,} trades, PF {s['profit_factor']:.2f}, expectancy {s['expectancy']*1e4:+.2f} bps")
    print(f"  VALIDATION: {v['trades']:,} trades, PF {v['profit_factor']:.2f}, expectancy {v['expectancy']*1e4:+.2f} bps")
    if v["expectancy"] > 0 and v["profit_factor"] > 1.0:
        print("\n  -> Holds up out-of-sample. Worth adopting, and re-checking as data accumulates.")
    else:
        print("\n  -> Does NOT hold out-of-sample: the selection-segment result was noise.")
        print("     Adopting it would be fitting the backtest, not finding an edge.")


if __name__ == "__main__":
    main()
