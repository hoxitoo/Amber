from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from amber.features.online import FeatureEngine


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def compute_batch_features(raw_root: Path, out_root: Path, symbols: list[str]) -> dict[str, int]:
    out_root.mkdir(parents=True, exist_ok=True)
    written = 0

    for symbol in symbols:
        source = raw_root / "normalized" / symbol / "part-000.jsonl"
        rows = _read_jsonl(source)
        if not rows:
            continue

        engine = FeatureEngine()
        feature_rows = engine.transform_rows(rows)
        feature_row = feature_rows[-1]

        target = out_root / "features" / symbol
        target.mkdir(parents=True, exist_ok=True)
        with (target / "part-000.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(feature_row, ensure_ascii=False) + "\n")
        written += 1

    return {"written_rows": written}
