"""Bounded JSONL reads.

Feature and normalized files grow for as long as the box runs — at 47k rows per
symbol, slurping one to look at its tail costs ~65ms and holds every row in
memory. The scanner does that for 20 symbols every minute, so the cost climbs
with accumulated history exactly like the feature recompute did. These helpers
stream instead, keeping only what the caller asked for.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_BLOCK = 1 << 16


def read_tail(path: Path, max_lines: int) -> list[dict[str, Any]]:
    """Parse at most the last `max_lines` JSON objects, skipping malformed ones.

    Reads backwards from the end of the file in blocks, so the cost is set by
    `max_lines`, not by the file. It used to stream the whole file through a
    deque: the scanner, the quality report and the forward ledger all ask for a
    few dozen to a few thousand rows of files that only ever grow (one per
    symbol, months long), and each call paid for the full history.

    Same line semantics as iterating the file: a final line without a trailing
    newline still counts, blank and malformed lines count toward `max_lines`
    and are then skipped.
    """
    if max_lines <= 0 or not Path(path).exists():
        return []
    with Path(path).open("rb") as fh:
        fh.seek(0, 2)
        pos = fh.tell()
        buf = b""
        while pos > 0 and buf.count(b"\n") <= max_lines:
            step = min(_BLOCK, pos)
            pos -= step
            fh.seek(pos)
            buf = fh.read(step) + buf
    lines = buf.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()  # the file ended with a newline: no line after it
    if pos > 0:
        lines = lines[1:]  # the first piece started mid-line
    out: list[dict[str, Any]] = []
    for raw in lines[-max_lines:]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except (json.JSONDecodeError, UnicodeDecodeError):
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
