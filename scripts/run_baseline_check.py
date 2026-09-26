"""Is the model a model, or a rebadged indicator? (roadmap D11)

`range_atr_14` carries 85% of permutation importance and live signals fire at
`range_atr_14=+4.2` — when volatility is already four sigma high. This ranks the
same rows by one raw feature, alerts on the same number of them, and reports how
many of the model's alerts the trivial rule also picks.

Overlap is the headline, not precision: it involves no outcomes, so a short test
segment with few independent episodes does not weaken it.

Read-only; writes `logs/baseline_comparison.json`.

    python scripts/run_baseline_check.py
    python scripts/run_baseline_check.py --rate 100 --features range_atr_14,bb_width_20
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.backtest.baselines import DEFAULT_BASELINES, compare_to_baselines, format_report, save_report  # noqa: E402
from amber.common.config import ConfigLoader, enter_project_root  # noqa: E402
from amber.common.logging import setup_logging  # noqa: E402


def _check_deps() -> str | None:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return (
            "Dependencies are missing from this interpreter.\n"
            f"  running: {sys.executable}\n\n"
            "Amber's services run from the project venv. Use it:\n"
            "  sudo -u amber /opt/amber/.venv/bin/python scripts/run_baseline_check.py\n\n"
            "Run as the `amber` user so logs/ stays writable by the pipeline."
        )
    return None


def main() -> int:
    problem = _check_deps()
    if problem:
        print(problem, file=sys.stderr)
        return 1

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rate", type=float, default=400.0, help="alerts/day to compare at")
    p.add_argument("--target", default="move", choices=("move", "pump", "dump"))
    p.add_argument("--features", default=",".join(DEFAULT_BASELINES), help="comma-separated feature names")
    args = p.parse_args()

    root = enter_project_root(__file__)
    config = ConfigLoader(root).load_yaml("config/amber.yaml")
    setup_logging(config.get("run", {}).get("log_level", "INFO"))
    storage = config.get("storage", {})

    report = compare_to_baselines(
        Path(storage["datasets_dir"]),
        Path(storage["models_dir"]),
        target=args.target,
        rate_per_day=args.rate,
        features=[f.strip() for f in args.features.split(",") if f.strip()],
    )
    print(format_report(report))
    if report.get("status") == "ok":
        print(f"\nsaved: {save_report(Path(storage['logs_dir']), report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
