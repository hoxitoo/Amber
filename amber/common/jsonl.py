"""Bounded JSONL reads.

Feature and normalized files grow for as long as the box runs — at 47k rows per
symbol, slurping one to look at its tail costs ~65ms and holds every row in
memory. The scanner does that for 20 symbols every minute, so the cost climbs
with accumulated history exactly like the feature recompute did. These helpers
stream instead, keeping only what the caller asked for.
"""

from __future__ import annotations

from collections import deque
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_tail(path: Path, max_lines: int) -> list[dict[str, Any]]:
    """Parse at most the last `max_lines` JSON objects, skipping malformed ones."""
    if max_lines <= 0 or not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8") as fh:
        lines = deque(fh, maxlen=max_lines)
    out: list[dict[str, Any]] = []
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return out


def read_last(path: Path) -> dict[str, Any] | None:
    """The newest valid JSON object, or None.

    Scans back from the end so a truncated final line (a writer killed mid-line)
    still yields the last complete record.
    """
    for row in reversed(read_tail(Path(path), 8)):
        return row
    return None
