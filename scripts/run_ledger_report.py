"""Print the forward ledger: would acting on the alerts have made money?

Read-only. Every live alert (model) and every shadow alert of the one-feature
rule `range_atr_14` is scored after its horizon by amber-pipeline; this only
summarises what has accumulated in logs/ledger.jsonl.

    python scripts/run_ledger_report.py
    python scripts/run_ledger_report.py --since 2026-10-01
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader, enter_project_root  # noqa: E402
from amber.monitoring.ledger import format_summary, summarize_ledger  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--since", default=None, help="UTC date YYYY-MM-DD; only alerts from then on")
    args = p.parse_args()

    root = enter_project_root(__file__)
    config = ConfigLoader(root).load_yaml("config/amber.yaml")
    since_ms = None
    if args.since:
        dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        since_ms = int(dt.timestamp() * 1000)
    summary = summarize_ledger(Path(config["storage"]["logs_dir"]), since_ms=since_ms)
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
