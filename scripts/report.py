from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
from amber.monitoring.reporting import build_system_report


def main() -> None:
    cfg = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    mon = cfg.get("monitoring", {}) if isinstance(cfg.get("monitoring", {}), dict) else {}
    freshness_sec = int(mon.get("model_eval_fresh_sec", 6 * 60 * 60))
    require_eval = bool(mon.get("require_model_eval_for_overall_ok", True))
    report = build_system_report(
        cfg["storage"],
        model_eval_fresh_sec=freshness_sec,
        require_model_eval_for_overall_ok=require_eval,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
