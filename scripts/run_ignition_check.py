"""Can a sharp move be predicted while the market is still quiet?

Read-only. Keeps only calm bars, labels whether a 1% move follows, trains in
memory and compares the model with every single feature as a rule. Writes only
logs/ignition_check.json. See amber/backtest/ignition.py for the definitions.

Holds the retrain lock while it runs, so the hourly retrain skips one cycle
instead of running beside it: both at once do not fit in the box's memory.

    python scripts/run_ignition_check.py
    python scripts/run_ignition_check.py --days 45
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader, enter_project_root  # noqa: E402
from amber.common.logging import setup_logging  # noqa: E402


def _check_deps() -> str | None:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return (
            "Dependencies are missing from this interpreter.\n"
            f"  running: {sys.executable}\n\n"
            "Use the project venv, as the amber user:\n"
            "  cd /opt/amber && sudo -u amber /opt/amber/.venv/bin/python scripts/run_ignition_check.py"
        )
    return None


def main() -> int:
    problem = _check_deps()
    if problem:
        print(problem, file=sys.stderr)
        return 1

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=30.0,
                    help="history to read per symbol (default 30; memory measured at 30)")
    args = ap.parse_args()
    root = enter_project_root(__file__)
    config = ConfigLoader(root).load_yaml("config/amber.yaml")
    setup_logging(config.get("run", {}).get("log_level", "INFO"))

    from amber.backtest.ignition import format_report, run_ignition_check, save_report
    from amber.common.locks import AlreadyRunning, SingleInstanceLock

    storage = config["storage"]
    labeling = config.get("labeling", {})
    split = config.get("model", {}).get("split", {})
    models_root = Path(storage["models_dir"])
    models_root.mkdir(parents=True, exist_ok=True)
    try:
        with SingleInstanceLock(models_root, "train"):
            report = run_ignition_check(
                Path(storage["features_dir"]),
                # Far longer than the 72h training window on purpose: moves
                # from a calm market are rare, and the window's test segment
                # held only ~10 of them (first live run, 2026-10-04).
                max_candles_per_symbol=int(args.days * 1440),
                min_warmup_bars=int(labeling.get("min_warmup_bars", 60)),
                train_frac=float(split.get("train_frac", 0.6)),
                calib_frac=float(split.get("calib_frac", 0.15)),
            )
    except AlreadyRunning:
        print(
            "Сейчас идёт переобучение модели (amber-pipeline). Вместе они не помещаются в память.\n"
            "Повтори через 3-4 минуты.",
            file=sys.stderr,
        )
        return 3
    print(format_report(report))
    if report.get("status") == "ok":
        print(f"\nsaved: {save_report(Path(storage['logs_dir']), report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
