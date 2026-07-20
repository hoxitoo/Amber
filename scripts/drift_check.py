from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
from amber.monitoring.drift import detect_drift


if __name__ == "__main__":
    config = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    features_root = Path(config["storage"]["features_dir"])
    models_root = Path(config["storage"]["models_dir"])
    symbols = config["exchange"]["bybit"]["symbols"]

    reference = None
    try:
        from amber.models.infer import load_latest_model

        ref = load_latest_model(models_root).get("train_reference")
        if isinstance(ref, dict) and ref:
            reference = ref
    except Exception:
        pass

    report = {s: detect_drift(features_root, s, reference=reference) for s in symbols}
    print(report)
