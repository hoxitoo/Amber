from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
from amber.models.calibrate import calibrate_model
from amber.models.eval import evaluate_model
from amber.models.registry import register_model
from amber.models.train import train_model
from amber.monitoring.metrics import emit_metrics


if __name__ == "__main__":
    config = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    datasets_root = Path(config["storage"]["datasets_dir"])
    models_root = Path(config["storage"]["models_dir"])
    logs_root = Path(config["storage"]["logs_dir"])

    tr = train_model(datasets_root=datasets_root, models_root=models_root)
    reg1 = register_model(models_root=models_root, model_run_id=tr["run_id"])

    cal = calibrate_model(models_root=models_root, datasets_root=datasets_root)
    reg2 = register_model(models_root=models_root, model_run_id=tr["run_id"], calibration_run_id=cal["run_id"])

    ev = evaluate_model(models_root=models_root, datasets_root=datasets_root)
    emit_metrics(logs_root, "model_precision_at_threshold", ev["precision_at_threshold"], {"model_run_id": tr["run_id"]})
    emit_metrics(logs_root, "model_brier", ev["brier"], {"model_run_id": tr["run_id"]})

    print({"train": tr, "register_pre_calib": reg1, "calibration": cal, "register_post_calib": reg2, "eval": ev})
