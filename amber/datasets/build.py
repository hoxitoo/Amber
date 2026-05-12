from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from amber.common.manifest import ArtifactManifest, new_run_id, write_manifest
from amber.labeling.events import label_event_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dataset(
    features_root: Path,
    datasets_root: Path,
    symbols: list[str],
    horizon_steps: int = 5,
    up_pct: float = 0.002,
    down_pct: float = 0.002,
) -> dict[str, int]:
    run_id = new_run_id(prefix="dataset")
    out_rows: list[dict[str, Any]] = []

    for symbol in symbols:
        feat_file = features_root / "features" / symbol / "part-000.jsonl"
        rows = _read_jsonl(feat_file)
        if not rows:
            continue

        prices = [float(r.get("mid_price", 0.0)) for r in rows]
        for i in range(len(rows)):
            future = prices[i : i + horizon_steps + 1]
            labels = label_event_path(future, up_pct=up_pct, down_pct=down_pct)
            out_rows.append(
                {
                    "symbol": symbol,
                    "ts": rows[i]["ts"],
                    "ret_1": rows[i].get("ret_1", 0.0),
                    "vol_z_20": rows[i].get("vol_z_20", 0.0),
                    "obs": rows[i].get("obs", 0),
                    "up_hit": labels["up_hit"],
                    "down_hit": labels["down_hit"],
                    "first_hit": labels["first_hit"],
                    "tte_idx": labels["tte_idx"],
                    "horizon_steps": horizon_steps,
                    "up_pct": up_pct,
                    "down_pct": down_pct,
                }
            )

    dataset_dir = datasets_root / run_id
    _write_jsonl(dataset_dir / "dataset.jsonl", out_rows)

    manifest = ArtifactManifest(
        run_id=run_id,
        artifact_type="dataset",
        artifact_version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        config_ref="config/amber.yaml",
        feature_spec_ref="config/features.yaml",
        metadata={
            "rows": len(out_rows),
            "symbols": symbols,
            "horizon_steps": horizon_steps,
            "up_pct": up_pct,
            "down_pct": down_pct,
        },
    )
    write_manifest(dataset_dir / "manifest.json", manifest)
    return {"rows": len(out_rows), "run_id": run_id}
