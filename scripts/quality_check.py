from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
from amber.monitoring.quality_report import build_quality_report


if __name__ == "__main__":
    cfg = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    logs = Path(cfg["storage"]["logs_dir"]) / "signals.jsonl"
    print(build_quality_report(logs))
