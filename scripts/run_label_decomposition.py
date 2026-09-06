"""Split the measured edge into magnitude and direction (roadmap D1).

The production label merges two questions the project states separately:
"catch volatility early" and "determine direction". Since
P(up_hit) = P(move) x P(up | move), lift on up_hit can come from either factor,
and a pure volatility forecaster scores lift ~2 on it by construction.

This trains one head per factor and reports each separately. Read-only: it
writes nothing but `logs/label_decomposition.json`.

    python scripts/run_label_decomposition.py
    python scripts/run_label_decomposition.py --horizon 30 --budget 0.02
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.backtest.decomposition import decompose, format_report, save_report  # noqa: E402
from amber.common.config import ConfigLoader  # noqa: E402
from amber.common.logging import setup_logging  # noqa: E402


def _check_deps() -> str | None:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return (
            "Dependencies are missing from this interpreter.\n"
            f"  running: {sys.executable}\n\n"
            "Amber's services run from the project venv. Use it:\n"
            "  sudo -u amber /opt/amber/.venv/bin/python scripts/run_label_decomposition.py\n\n"
            "Run as the `amber` user so logs/ stays writable by the pipeline."
        )
    return None


def main() -> int:
    problem = _check_deps()
    if problem:
        print(problem, file=sys.stderr)
        return 1

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizon", type=int, default=None, help="bars (default: config horizon_steps)")
    p.add_argument("--barrier", type=float, default=None, help="fraction, e.g. 0.010 (default: config up_pct)")
    p.add_argument("--budget", type=float, default=0.01, help="alert budget as a fraction of scored rows")
    p.add_argument("--max-candles", type=int, default=None, help="candles per symbol (default: config)")
    args = p.parse_args()

    root = Path.cwd()
    config = ConfigLoader(root).load_yaml("config/amber.yaml")
    setup_logging(config.get("run", {}).get("log_level", "INFO"))
    lab = config.get("labeling", {})
    storage = config.get("storage", {})

    report = decompose(
        Path(storage["features_dir"]),
        horizon=int(args.horizon if args.horizon is not None else lab.get("horizon_steps", 15)),
        barrier=float(args.barrier if args.barrier is not None else lab.get("up_pct", 0.010)),
        budget=args.budget,
        max_candles_per_symbol=int(
            args.max_candles if args.max_candles is not None else lab.get("max_candles_per_symbol", 2880)
        ),
        min_warmup_bars=int(lab.get("min_warmup_bars", 60)),
    )
    print(format_report(report))
    if report.get("status") == "ok":
        print(f"\nsaved: {save_report(Path(storage['logs_dir']), report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
