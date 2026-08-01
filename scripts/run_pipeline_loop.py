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
from amber.common.retention import cleanup_consumed_ws_raw, dataset_keep, free_bytes, prune_run_dirs
from amber.datasets.build import build_dataset_from_config
from amber.pipeline import collector_app, features_app, normalize_app
from amber.pipeline.normalize_app import OFFSETS_STATE_KEY
from amber.pipeline.train_app import NotEnoughData, run_training
from amber.storage.state_store import StateStore

logger = logging.getLogger(__name__)

# Below this much free space the box is one flush away from ENOSPC, which used to
# wedge every stage at once. Sweep hard and keep fewer runs until it recovers.
LOW_DISK_BYTES = 1_500_000_000  # 1.5 GB


def _retention_sweep(config: dict) -> None:
    """Reclaim disk at the START of every cycle.

    Retention used to run only as the tail of a successful training run, so the
    moment the disk filled (training fails first) nothing was ever pruned again —
    the cleanup was unreachable exactly when it was needed. Running it up front,
    unconditionally, is what lets the box dig itself out (audit C4).
    """
    storage = config.get("storage", {})
    try:
        datasets_root = Path(storage["datasets_dir"])
        models_root = Path(storage["models_dir"])
        raw_root = Path(storage["raw_dir"])
        state = StateStore(Path(storage["state_dir"]))
    except (KeyError, TypeError):
        return

    keep = int(storage.get("keep_runs", 5) or 5)
    keep_datasets = dataset_keep(storage)
    low = free_bytes(raw_root) < LOW_DISK_BYTES
    if low:
        keep = keep_datasets = 1
        logger.warning("low disk (%.2f GB free): pruning to the newest run only", free_bytes(raw_root) / 1e9)

    try:
        offsets = dict(state.get(OFFSETS_STATE_KEY))
        deleted, freed = cleanup_consumed_ws_raw(raw_root, offsets)
        if deleted:
            state.set(OFFSETS_STATE_KEY, offsets)
        for root, prefix, n in (
            (datasets_root, "dataset_", keep_datasets),
            (models_root, "model_", keep),
            (models_root, "calib_", keep),
        ):
            d, f = prune_run_dirs(root, prefix, n)
            deleted += d
            freed += f
        if deleted:
            logger.info("retention sweep freed %.1f MB (%s items)", freed / 1e6, deleted)
    except Exception as exc:  # never let housekeeping kill the loop
        logger.error("retention sweep failed: %s", exc)


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

    # Free space before doing anything else: if the box was wedged by a full
    # disk, a restart must be able to dig out instead of immediately re-failing.
    _retention_sweep(cfg)

    # Seed history on startup so a fresh box has enough candles to train within
    # minutes instead of waiting ~1h for the WS stream. Idempotent (watermark
    # dedup), so a restart never duplicates data.
    try:
        collector_app.main()
    except Exception as exc:
        logger.warning("startup backfill failed (will rely on WS stream): %s", exc)

    # Retraining is by far the heaviest step. If it is what exhausted memory,
    # restarting straight into another one turns a single OOM into a crash loop,
    # so only a box with no model yet retrains immediately; one that already has
    # a model waits out the normal interval and keeps collecting meanwhile.
    models_dir = Path(cfg.get("storage", {}).get("models_dir", "data/models"))
    has_model = any(models_dir.glob("model_*/model.json")) if models_dir.exists() else False
    last_retrain = time.time() if has_model else 0.0
    if has_model:
        logger.info("existing model found; first retrain in %s min (restart-safe)", retrain_min)

    while True:
        _retention_sweep(cfg)

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
