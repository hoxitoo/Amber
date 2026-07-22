from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
from amber.models.calibrate import calibrate_model
from amber.models.dataset_io import load_latest_dataset_rows
from amber.models.eval import evaluate_model
from amber.models.registry import register_model
from amber.models.train import train_model
from amber.monitoring.metrics import emit_metrics

# Below this many usable rows a model cannot be trained honestly (warm-up gating
# needs >= min_warmup_bars candles per symbol before any row qualifies).
MIN_TRAIN_ROWS = 50


def _guard_enough_data(datasets_root: Path) -> None:
    try:
        rows, _run = load_latest_dataset_rows(datasets_root)
    except ValueError:
        rows = []
    if len(rows) >= MIN_TRAIN_ROWS:
        return
    print(
        "Недостаточно данных для обучения.\n"
        f"В датасете {len(rows)} строк (нужно >= {MIN_TRAIN_ROWS}).\n\n"
        "Что делать:\n"
        "  1) Нажми «REST-backfill истории» — это мгновенно подтянет ~200-1000 свечей\n"
        "     на символ, чтобы преодолеть порог warm-up (>= 60 свечей на символ).\n"
        "  2) Затем «Нормализация» → «Фичи» → «Датасет» (или снова «Полный цикл»).\n"
        "  3) Либо просто дай WS-коллектору поработать 1-2 часа и повтори.\n\n"
        "Это не ошибка — модели пока не на чем учиться.",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    config = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    datasets_root = Path(config["storage"]["datasets_dir"])
    models_root = Path(config["storage"]["models_dir"])
    logs_root = Path(config["storage"]["logs_dir"])

    _guard_enough_data(datasets_root)

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
    reg1 = register_model(models_root=models_root, model_run_id=tr["run_id"])

    cal = calibrate_model(
        models_root=models_root,
        datasets_root=datasets_root,
        holdout_ratio=float(cal_cfg.get("holdout_ratio", 0.2)),
    )
    reg2 = register_model(models_root=models_root, model_run_id=tr["run_id"], calibration_run_id=cal["run_id"])

    ev = evaluate_model(
        models_root=models_root,
        datasets_root=datasets_root,
        threshold=float(eval_cfg.get("threshold", 0.7)),
    )
    emit_metrics(logs_root, "model_precision_at_threshold", ev["precision_at_threshold"], {"model_run_id": tr["run_id"]})
    emit_metrics(logs_root, "model_precision_up_at_threshold", ev["precision_up_at_threshold"], {"model_run_id": tr["run_id"]})
    emit_metrics(logs_root, "model_precision_down_at_threshold", ev["precision_down_at_threshold"], {"model_run_id": tr["run_id"]})
    emit_metrics(logs_root, "model_brier", ev["brier"], {"model_run_id": tr["run_id"]})
    emit_metrics(logs_root, "model_brier_up_cal", ev["brier_up_cal"], {"model_run_id": tr["run_id"]})
    emit_metrics(logs_root, "model_brier_down_cal", ev["brier_down_cal"], {"model_run_id": tr["run_id"]})
    for key in ("pr_auc_up_cal", "pr_auc_down_cal", "pr_auc_up_lift", "pr_auc_down_lift"):
        if key in ev:
            emit_metrics(logs_root, f"model_{key}", ev[key], {"model_run_id": tr["run_id"]})

    print({"train": tr, "register_pre_calib": reg1, "calibration": cal, "register_post_calib": reg2, "eval": ev})
