from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
from amber.datasets.build import build_dataset_from_config


if __name__ == "__main__":
    from amber.common.locks import AlreadyRunning

    config = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    try:
        print(build_dataset_from_config(config))
    except AlreadyRunning:
        print("Датасет уже собирается (amber-pipeline по расписанию). Повтори через минуту.", file=sys.stderr)
        raise SystemExit(3) from None
