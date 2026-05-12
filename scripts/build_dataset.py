from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
from amber.datasets.build import build_dataset


if __name__ == "__main__":
    config = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    symbols = config["exchange"]["bybit"]["symbols"]
    result = build_dataset(
        features_root=Path(config["storage"]["features_dir"]),
        datasets_root=Path(config["storage"]["datasets_dir"]),
        symbols=symbols,
    )
    print(result)
