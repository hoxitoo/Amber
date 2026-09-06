"""Precision at an alert rate a person can act on (roadmap D4).

Everything measured before this used a 1% alert budget — ~388 alerts/day across
27 symbols, which is a firehose rather than a tool. This sweeps the alert rate
over the live model's out-of-sample segment and reports the calibrated
probability cut, the equivalent `prob_lift_min`, and the precision each rate
buys, so the live threshold can be set from measurement instead of habit.

Read-only: uses the existing model and calibration artifacts and writes only
`logs/operating_curve.json`.

    python scripts/run_operating_curve.py
    python scripts/run_operating_curve.py --rates 100,50,20,10 --target pump
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.backtest.operating_point import DEFAULT_RATES, format_curve, operating_curve, save_curve  # noqa: E402
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
            "  sudo -u amber /opt/amber/.venv/bin/python scripts/run_operating_curve.py\n\n"
            "Run as the `amber` user so logs/ stays writable by the pipeline."
        )
    return None


def main() -> int:
    problem = _check_deps()
    if problem:
        print(problem, file=sys.stderr)
        return 1

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--rates",
        default=",".join(str(int(r)) for r in DEFAULT_RATES),
        help="comma-separated alerts/day to evaluate",
    )
    p.add_argument("--target", default="move", choices=("move", "pump", "dump"))
    args = p.parse_args()

    root = Path.cwd()
    config = ConfigLoader(root).load_yaml("config/amber.yaml")
    setup_logging(config.get("run", {}).get("log_level", "INFO"))
    storage = config.get("storage", {})

    report = operating_curve(
        Path(storage["datasets_dir"]),
        Path(storage["models_dir"]),
        target=args.target,
        rates_per_day=[float(x) for x in args.rates.split(",") if x.strip()],
    )
    print(format_curve(report))
    if report.get("status") == "ok":
        print(f"\nsaved: {save_curve(Path(storage['logs_dir']), report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
