"""Shared training pipeline: train -> calibrate -> eval -> register + metrics.

Used both by scripts/train_model.py and by the auto-pipeline loop so there is a
single source of truth for how a model is produced.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from amber.common.retention import dataset_keep, prune_run_dirs
from amber.models.calibrate import calibrate_model
from amber.models.dataset_io import load_latest_dataset_rows
from amber.models.eval import evaluate_model
from amber.models.registry import register_model
from amber.models.train import train_model
from amber.monitoring.metrics import emit_metrics

logger = logging.getLogger(__name__)

# Below this many usable rows a model cannot be trained honestly (warm-up gating
# needs >= min_warmup_bars candles per symbol before any row qualifies).
MIN_TRAIN_ROWS = 50

_METRIC_KEYS = (
    "precision_at_threshold",
    "precision_up_at_threshold",
    "precision_down_at_threshold",
    "brier",
    "brier_up_cal",
    "brier_down_cal",
    "pr_auc_up_cal",
    "pr_auc_down_cal",
    "pr_auc_up_lift",
    "pr_auc_down_lift",
)


class NotEnoughData(Exception):
    """Raised when the dataset is too small to train an honest model."""

    def __init__(self, rows: int) -> None:
        self.rows = rows
        super().__init__(f"dataset has {rows} rows (< {MIN_TRAIN_ROWS})")


def run_training(
    config: dict[str, Any],
    datasets_root: Path,
    models_root: Path,
    logs_root: Path,
) -> dict[str, Any]:
    """Train + calibrate + evaluate + register the latest dataset.

    Raises NotEnoughData when the dataset is too small (callers decide whether
    that is a hard error or an informational skip).
    """
    try:
        rows, _run = load_latest_dataset_rows(datasets_root)
    except ValueError:
        rows = []
    if len(rows) < MIN_TRAIN_ROWS:
        raise NotEnoughData(len(rows))

    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    cv_cfg = model_cfg.get("cv", {})
    cal_cfg = model_cfg.get("calibration", {})
    eval_cfg = model_cfg.get("eval", {})
    split_cfg = model_cfg.get("split", {})

    tr = train_model(
        datasets_root=datasets_root,
        models_root=models_root,
        cv_n_folds=int(cv_cfg.get("n_folds", 3)),
        cv_val_size=int(cv_cfg.get("val_size", 50)),
        cv_gap_candles=int(cv_cfg.get("gap_candles", 30)),
        train_frac=float(split_cfg.get("train_frac", 0.7)),
        calib_frac=float(split_cfg.get("calib_frac", 0.15)),
    )
    register_model(models_root=models_root, model_run_id=tr["run_id"])
    cal = calibrate_model(
        models_root=models_root,
        datasets_root=datasets_root,
        holdout_ratio=float(cal_cfg.get("holdout_ratio", 0.2)),
    )
    register_model(models_root=models_root, model_run_id=tr["run_id"], calibration_run_id=cal["run_id"])
    ev = evaluate_model(
        models_root=models_root,
        datasets_root=datasets_root,
        threshold=float(eval_cfg.get("threshold", 0.7)),
    )
    for key in _METRIC_KEYS:
        if key in ev:
            emit_metrics(logs_root, f"model_{key}", ev[key], {"model_run_id": tr["run_id"]})

    # Publish the promotion-gate backtest here, where a full dataset load is
    # expected, so the dashboard can render it without recomputing per page view.
    try:
        from amber.backtest.backtester import event_backtest
        from amber.monitoring.reporting import _load_thresholds, save_backtest_result

        storage_paths = {k: str(v) for k, v in config.get("storage", {}).items() if isinstance(v, str)}
        bt = event_backtest(datasets_root, models_root, thresholds=_load_thresholds(storage_paths))
        save_backtest_result(logs_root, bt)
    except Exception as exc:  # a failed replay must not fail the retrain
        logger.warning("backtest publish failed: %s", exc)

    # Retention: hourly auto-retrain would otherwise pile up dataset/model dirs
    # forever. Keep only the most recent runs.
    storage_cfg = config.get("storage", {})
    keep = int(storage_cfg.get("keep_runs", 5))
    prune_run_dirs(datasets_root, "dataset_", dataset_keep(storage_cfg))
    prune_run_dirs(models_root, "model_", keep)
    prune_run_dirs(models_root, "calib_", keep)
    return {"train": tr, "calibration": cal, "eval": ev}
