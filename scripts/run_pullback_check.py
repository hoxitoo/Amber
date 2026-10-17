"""Pullback-to-support on 4H bars, on years of history. Read-only study.

Downloads public candles and funding (no keys), runs the twelve pre-registered
variants of amber/backtest/pullback.py on BTC and ETH, and prints which, if
any, make money after costs and beat a random entry in the same trend.

    python scripts/run_pullback_check.py                    # Bybit (run on the VPS)
    python scripts/run_pullback_check.py --source binance   # public archive, cached

Touches none of the running services' files; Binance archives are cached in
data/history/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader, enter_project_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("bybit", "binance"), default="bybit")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    args = ap.parse_args()

    root = enter_project_root(__file__)
    config = ConfigLoader(root).load_yaml("config/amber.yaml")
    from amber.backtest.history import load_history
    from amber.backtest.pullback import format_study, run_study

    cache = Path(config["storage"].get("root", "data")) / "history"
    series = {}
    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        bars, funding = load_history(args.source, sym, "4h", cache)
        print(f"{sym}: {len(bars)} bars, {len(funding)} funding rates", flush=True)
        series[sym] = (bars, funding)
    result = run_study(series)
    print()
    print(format_study(result, args.source))
    out = Path(config["storage"]["logs_dir"]) / f"pullback_{args.source}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
