"""Continuously turn accumulating raw WS data into fresh features, and
periodically rebuild the dataset + retrain the model.

The WS collector only *accumulates* raw payloads. This loop:
  - every `loop_sec`: normalize -> features (keeps candles/features current);
  - every `retrain_min`: build_dataset -> train (so the dataset and model grow
    with the accumulating data without any manual clicking).
Start/stop it from the dashboard control panel.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
from amber.common.logging import setup_logging
from amber.datasets.build import build_dataset_from_config
from amber.pipeline import collector_app, features_app, normalize_app
from amber.pipeline.train_app import NotEnoughData, run_training

logger = logging.getLogger(__name__)


def _retrain(config: dict) -> None:
    build_dataset_from_config(config)
    result = run_training(
        config,
        Path(config["storage"]["datasets_dir"]),
        Path(config["storage"]["models_dir"]),
        Path(config["storage"]["logs_dir"]),
    )
    ev = result["eval"]
    logger.info(
        "auto-retrain done model=%s pr_auc_up=%s in_sample=%s",
        result["train"]["model_type"],
        ev.get("pr_auc_up_cal"),
        ev.get("in_sample"),
    )


def main() -> None:
    cfg = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    setup_logging(cfg.get("run", {}).get("log_level", "INFO"))
    pipeline_cfg = cfg.get("pipeline", {}) if isinstance(cfg.get("pipeline", {}), dict) else {}
    interval = max(15, int(pipeline_cfg.get("loop_sec", 60)))
    retrain_min = int(pipeline_cfg.get("retrain_min", 60))
    logger.info("pipeline loop started interval=%ss retrain_min=%s", interval, retrain_min)

    # Seed history on startup so a fresh box has enough candles to train within
    # minutes instead of waiting ~1h for the WS stream. Idempotent (watermark
    # dedup), so a restart never duplicates data.
    try:
        collector_app.main()
    except Exception as exc:
        logger.warning("startup backfill failed (will rely on WS stream): %s", exc)

    last_retrain = 0.0
    while True:
        for stage, fn in (("normalize", normalize_app.main), ("features", features_app.main)):
            try:
                fn()
            except Exception as exc:  # keep the loop alive across transient errors
                logger.error("pipeline stage %s failed: %s", stage, exc)

        now = time.time()
        if retrain_min > 0 and (now - last_retrain) >= retrain_min * 60:
            try:
                _retrain(cfg)
            except NotEnoughData as exc:
                logger.info("auto-retrain skipped: %s", exc)
            except Exception as exc:
                logger.error("auto-retrain failed: %s", exc)
            last_retrain = now

        time.sleep(interval)


if __name__ == "__main__":
    main()
