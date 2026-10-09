"""Train the ignition model now instead of waiting for the daily run.

Holds the retrain lock, so it never runs beside the hourly retrain.

    python scripts/train_ignition.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader, enter_project_root  # noqa: E402
from amber.common.logging import setup_logging  # noqa: E402


def main() -> int:
    root = enter_project_root(__file__)
    config = ConfigLoader(root).load_yaml("config/amber.yaml")
    setup_logging(config.get("run", {}).get("log_level", "INFO"))
    from amber.common.locks import AlreadyRunning, SingleInstanceLock
    from amber.signals.ignition_live import train_ignition

    ign = config.get("ignition", {})
    models_root = Path(config["storage"]["models_dir"])
    models_root.mkdir(parents=True, exist_ok=True)
    try:
        with SingleInstanceLock(models_root, "train"):
            res = train_ignition(
                Path(config["storage"]["features_dir"]), models_root,
                days=float(ign.get("train_days", 30)),
                min_warmup_bars=int(config.get("labeling", {}).get("min_warmup_bars", 60)),
                budget=float(ign.get("budget", 0.01)),
            )
    except AlreadyRunning:
        print("Сейчас идёт другое обучение (amber-pipeline). Повтори через 3-4 минуты.", file=sys.stderr)
        return 3
    print(res)
    return 0 if res.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
