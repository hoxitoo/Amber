"""Read-only sweep over label definitions.

Trains one model per arm in memory and reports precision at a fixed alert
budget, with and without the one-bar delay between the alert and acting on it.
Writes nothing except `logs/label_sweep.json`: no dataset is rebuilt and no
model is registered, so this is safe to run on a live box.

    python scripts/run_label_sweep.py
    python scripts/run_label_sweep.py --horizons 15,30,60,120 --budget 0.005
    python scripts/run_label_sweep.py --target dump
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.backtest.label_sweep import RULERS, SHAPES, format_table, run_sweep, save_report  # noqa: E402
from amber.common.config import ConfigLoader  # noqa: E402
from amber.common.logging import setup_logging  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizons", default="15,30,60", help="comma-separated bar counts")
    p.add_argument("--rulers", default=",".join(RULERS), help=f"any of: {', '.join(RULERS)}")
    p.add_argument("--shapes", default=",".join(SHAPES), help=f"any of: {', '.join(SHAPES)}")
    p.add_argument("--budget", type=float, default=0.002, help="alert budget as a fraction of scored rows")
    p.add_argument("--target", default="pump", choices=("pump", "dump"))
    p.add_argument("--max-candles", type=int, default=None, help="candles per symbol (default: config)")
    args = p.parse_args()

    root = Path.cwd()
    config = ConfigLoader(root).load_yaml("config/amber.yaml")
    setup_logging(config.get("run", {}).get("log_level", "INFO"))
    labeling = config.get("labeling", {})
    storage = config.get("storage", {})

    report = run_sweep(
        Path(storage["features_dir"]),
        horizons=[int(x) for x in args.horizons.split(",") if x.strip()],
        rulers=[x.strip() for x in args.rulers.split(",") if x.strip()],
        shapes=[x.strip() for x in args.shapes.split(",") if x.strip()],
        budget=args.budget,
        max_candles_per_symbol=int(
            args.max_candles if args.max_candles is not None else labeling.get("max_candles_per_symbol", 2880)
        ),
        min_warmup_bars=int(labeling.get("min_warmup_bars", 60)),
        k=float(labeling.get("threshold_k", 0.8)),
        floor=float(labeling.get("threshold_floor", 0.005)),
        cap=float(labeling.get("threshold_cap", 0.05)),
        target=args.target,
    )
    print(format_table(report))
    if report.get("status") == "ok":
        out = save_report(Path(storage["logs_dir"]), report)
        print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
