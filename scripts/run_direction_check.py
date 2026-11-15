"""Does the order book say which way a move will go? (roadmap D2) Read-only.

    python scripts/run_direction_check.py

Reads the book collector's rows (data/raw/book), the 1m candles and the
ignition ledger, and reports whether any book or flow feature, or a small
model on all of them, calls the side of the next 1% move better than always
calling the more frequent side. Rules and bounds: amber/backtest/direction.py.

Holds the retrain lock while it runs, so it never shares memory with a
retrain; one hourly retrain may be skipped.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader, enter_project_root  # noqa: E402


def main() -> int:
    root = enter_project_root(__file__)
    config = ConfigLoader(root).load_yaml("config/amber.yaml")
    from amber.backtest.direction import format_report, run_direction_check, save_report
    from amber.common.locks import AlreadyRunning, SingleInstanceLock

    storage = config["storage"]
    models_root = Path(storage["models_dir"])
    models_root.mkdir(parents=True, exist_ok=True)
    try:
        with SingleInstanceLock(models_root, "train"):
            report = run_direction_check(Path(storage["raw_dir"]), Path(storage["logs_dir"]),
                                         list(config["exchange"]["bybit"]["symbols"]))
    except AlreadyRunning:
        print("Сейчас идёт переобучение модели (amber-pipeline). Повтори через 3-4 минуты.", file=sys.stderr)
        return 3
    print(format_report(report))
    print(f"\nsaved: {save_report(Path(storage['logs_dir']), report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
