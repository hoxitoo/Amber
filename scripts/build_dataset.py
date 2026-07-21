from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
from amber.datasets.build import build_dataset


if __name__ == "__main__":
    config = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    symbols = config["exchange"]["bybit"]["symbols"]
    labeling = config.get("labeling", {})
    result = build_dataset(
        features_root=Path(config["storage"]["features_dir"]),
        datasets_root=Path(config["storage"]["datasets_dir"]),
        symbols=symbols,
        horizon_steps=int(labeling.get("horizon_steps", 5)),
        up_pct=float(labeling.get("up_pct", 0.002)),
        down_pct=float(labeling.get("down_pct", 0.002)),
        adaptive_thresholds=bool(labeling.get("adaptive_thresholds", False)),
        threshold_k=float(labeling.get("threshold_k", 2.5)),
        threshold_vol_window=int(labeling.get("threshold_vol_window", 60)),
        threshold_floor=float(labeling.get("threshold_floor", 0.003)),
        threshold_cap=float(labeling.get("threshold_cap", 0.05)),
        horizon_steps_list=[int(x) for x in labeling.get("horizon_steps_list", [])],
        min_warmup_bars=int(labeling.get("min_warmup_bars", 60)),
    )
    print(result)
