from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from amber.alerts.console import print_alert
from amber.common.config import ConfigLoader
from amber.common.logging import setup_logging
from amber.signals.filters import passes_thresholds
from amber.signals.scorer import score_signal

logger = logging.getLogger(__name__)


def _read_latest_feature_rows(features_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file in sorted((features_root / "features").glob("*/part-000.jsonl")):
        lines = file.read_text(encoding="utf-8").splitlines()
        if not lines:
            continue
        rows.append(json.loads(lines[-1]))
    return rows


def main() -> None:
    config = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    thresholds = ConfigLoader(Path.cwd()).load_yaml("config/thresholds.yaml")
    setup_logging(config.get("run", {}).get("log_level", "INFO"))

    features_root = Path(config["storage"]["features_dir"])
    models_root = Path(config["storage"]["models_dir"])

    feature_rows = _read_latest_feature_rows(features_root)
    emitted = 0
    for row in feature_rows:
        signal = score_signal(row, models_root=models_root, config_version=config["signal"]["schema_version"])
        if passes_thresholds(
            signal,
            up_min=float(thresholds["thresholds"]["pump_prob_calibrated_min"]),
            down_min=float(thresholds["thresholds"]["dump_prob_calibrated_min"]),
        ):
            print_alert(signal)
            emitted += 1

    logger.info("scanner finished symbols=%s emitted=%s", len(feature_rows), emitted)


if __name__ == "__main__":
    main()
