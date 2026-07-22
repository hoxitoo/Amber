from pathlib import Path
import json
import logging
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
from amber.monitoring.reporting import build_system_report
from amber.storage.state_store import StateStore

logger = logging.getLogger(__name__)


def _push_on_flip(report: dict, state: StateStore) -> None:
    """Push a Telegram note when overall readiness flips (audit A6): a silent
    collector death should reach the owner's phone, not wait to be discovered."""
    prev = state.get("report_status")
    current_ok = bool(report.get("overall_ok"))
    if prev and bool(prev.get("overall_ok")) != current_ok:
        from amber.alerts.telegram import send_telegram_text

        if current_ok:
            text = "AMBER: система снова в норме (overall_ok=true)."
        else:
            reason = report.get("overall_reason", "unknown")
            failed = ", ".join(report.get("readiness_failed_components", [])) or reason
            text = f"AMBER: система деградировала (overall_ok=false). Компоненты: {failed}."
        send_telegram_text(text)
        logger.warning("overall_ok flipped to %s", current_ok)
    state.set("report_status", {"overall_ok": current_ok, "reason": report.get("overall_reason")})


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
    _push_on_flip(report, StateStore(Path(cfg["storage"]["state_dir"])))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
