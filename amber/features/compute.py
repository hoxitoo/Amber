from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def _safe_ret(last_prices: list[float]) -> float:
    if len(last_prices) < 2:
        return 0.0
    prev = last_prices[-2]
    cur = last_prices[-1]
    if prev == 0:
        return 0.0
    return (cur / prev) - 1.0


def compute_batch_features(raw_root: Path, out_root: Path, symbols: list[str]) -> dict[str, int]:
    out_root.mkdir(parents=True, exist_ok=True)
    written = 0

    for symbol in symbols:
        source = raw_root / "ticks" / symbol / "part-000.jsonl"
        rows = _read_jsonl(source)
        if not rows:
            continue

        last_series = [float(r["last"]) for r in rows]
        volume_series = [float(r.get("volume_1m", 0.0)) for r in rows]

        feature_row = {
            "symbol": symbol,
            "ts": rows[-1]["ts"],
            "ret_1m": _safe_ret(last_series),
            "mid_price": (float(rows[-1]["bid"]) + float(rows[-1]["ask"])) / 2.0,
            "vol_mean": mean(volume_series),
            "obs": len(rows),
        }

        target = out_root / "features" / symbol
        target.mkdir(parents=True, exist_ok=True)
        with (target / "part-000.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(feature_row, ensure_ascii=False) + "\n")
        written += 1

    return {"written_rows": written}
