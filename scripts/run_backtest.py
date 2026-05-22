from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.backtest.backtester import event_backtest
from amber.common.config import ConfigLoader


if __name__ == "__main__":
    cfg = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    res = event_backtest(Path(cfg["storage"]["datasets_dir"]))
    print(json.dumps(res, indent=2, sort_keys=True))
