from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
from amber.monitoring.health import check_health


if __name__ == "__main__":
    config = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    root = Path(config["storage"]["root"])
    report = check_health(root)
    print({"ok": report.ok, "checks": report.checks})
