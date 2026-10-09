"""Ignition warnings, checked forward: do they precede moves?

Read-only. Prints the forward confirmation and the latest warnings a person
would have been shown. See amber/signals/ignition_live.py.

    python scripts/run_ignition_report.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader, enter_project_root  # noqa: E402


def main() -> int:
    root = enter_project_root(__file__)
    config = ConfigLoader(root).load_yaml("config/amber.yaml")
    from amber.common.jsonl import read_tail
    from amber.signals.ignition_live import RECORDS_FILE, alert_text, format_ignition_summary, summarize_ignition

    logs = Path(config["storage"]["logs_dir"])
    print(format_ignition_summary(summarize_ignition(logs)))
    shown = [r for r in read_tail(logs / RECORDS_FILE, 5000) if r.get("notify")][-10:]
    if shown:
        from datetime import datetime, timezone

        print("\nПоследние предупреждения:")
        for r in shown:
            when = datetime.fromtimestamp(r["event_ts"] / 1000, tz=timezone.utc).strftime("%m-%d %H:%M UTC")
            print(f"  {when}  {alert_text(r)}")
    elif (logs / RECORDS_FILE).exists():
        print("\nПредупреждений пока не было.")
    else:
        print("\nЗаписей ещё нет: модель обучается раз в сутки, сканер начнёт после первого обучения.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
