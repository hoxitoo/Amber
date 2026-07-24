from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ParquetSink:
    """JSONL fallback sink for early-stage local runs.

    Keeps interface stable; storage backend can be swapped to parquet writer later
    without changing upstream collector code.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_records(
        self,
        topic: str,
        symbol: str,
        records: list[dict[str, Any]],
        part: str = "part-000",
    ) -> Path:
        target = self.root / topic / symbol
        target.mkdir(parents=True, exist_ok=True)
        out = target / f"{part}.jsonl"
        with out.open("a", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return out
