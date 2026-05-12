from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def emit_metrics(logs_root: Path, metric_name: str, value: float, tags: dict[str, Any] | None = None) -> None:
    tags = tags or {}
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "metric": metric_name,
        "value": value,
        "tags": tags,
    }
    out = logs_root / "metrics.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
