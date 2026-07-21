"""Append-only audit trail of which symbols were in scope over time.

Records the universe membership per stage/run so selection is reproducible and
survivorship/selection bias can be inspected after the fact (audit finding Q1).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


def log_universe(logs_dir: Path, stage: str, symbols: list[str], extra: dict | None = None) -> None:
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "symbols": sorted(symbols),
        "count": len(symbols),
    }
    if extra:
        row.update(extra)
    with (logs_dir / "universe.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
