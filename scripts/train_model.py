from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
from amber.models.calibrate import calibrate_model
from amber.models.eval import evaluate_model
from amber.models.train import train_model


if __name__ == "__main__":
    config = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    datasets_root = Path(config["storage"]["datasets_dir"])
    models_root = Path(config["storage"]["models_dir"])

    tr = train_model(datasets_root=datasets_root, models_root=models_root)
    cal = calibrate_model(models_root=models_root, datasets_root=datasets_root)
    ev = evaluate_model(models_root=models_root, datasets_root=datasets_root)
    print({"train": tr, "calibration": cal, "eval": ev})
