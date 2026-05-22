from pathlib import Path
<<<<<<< HEAD
=======
import json
>>>>>>> origin/main
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
<<<<<<< HEAD
from amber.monitoring.quality_report import build_quality_report
=======
from amber.monitoring.drift import PredictionBiasMonitor, RollingAUCMonitor, psi_monitor
>>>>>>> origin/main


if __name__ == "__main__":
    cfg = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    logs = Path(cfg["storage"]["logs_dir"]) / "signals.jsonl"
<<<<<<< HEAD
    print(build_quality_report(logs))
=======

    auc_m = RollingAUCMonitor(window=200)
    bias_m = PredictionBiasMonitor(window=200)

    rows = []
    if logs.exists():
        rows = [json.loads(x) for x in logs.read_text(encoding="utf-8").splitlines() if x.strip()]

    # Local placeholder: use prob_up_calibrated as score; outcome unavailable in live yet.
    for r in rows:
        p_up = float(r.get("prob_up_calibrated", 0.0))
        p_dn = float(r.get("prob_down_calibrated", 0.0))
        bias_m.update(p_up, p_dn)
        # pseudo-outcome placeholder for local smoke only
        auc_m.update(int(p_up > p_dn), p_up)

    bias = bias_m.bias()
    auc = auc_m.value()

    # simple PSI check on up probs split by time
    up = [float(r.get("prob_up_calibrated", 0.0)) for r in rows]
    mid = len(up) // 2
    psi_result = psi_monitor(up[:mid], up[mid:]) if len(up) >= 20 else {"psi": 0.0, "level": "low"}

    print({"signals": len(rows), "rolling_auc": auc, "prediction_bias": bias, "psi": psi_result})
>>>>>>> origin/main
