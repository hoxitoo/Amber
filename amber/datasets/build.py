from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from amber.common.manifest import ArtifactManifest, new_run_id, write_manifest
from amber.labeling.events import label_event_path
from amber.models.features import MODEL_FEATURES


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {lineno}: {exc.msg}") from exc
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rolling_vol(values: list[float], end_idx: int, window: int) -> float:
    start = max(0, end_idx - window + 1)
    chunk = values[start : end_idx + 1]
    if not chunk:
        return 0.0
    mean = sum(chunk) / len(chunk)
    var = sum((x - mean) ** 2 for x in chunk) / len(chunk)
    return math.sqrt(max(0.0, var))


def _adaptive_threshold(
    ret_1: list[float],
    idx: int,
    k: float,
    window: int,
    floor: float,
    cap: float,
) -> float:
    vol = _rolling_vol(ret_1, end_idx=idx, window=window)
    return max(floor, min(cap, k * vol))


def _validate_horizons(horizon_steps: int, horizon_steps_list: list[int] | None) -> list[int]:
    if horizon_steps_list:
        horizons = sorted(set(horizon_steps_list))
        if any(h <= 0 for h in horizons):
            raise ValueError("all horizons must be > 0")
        return horizons
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be > 0")
    return [horizon_steps]


def build_dataset(
    features_root: Path,
    datasets_root: Path,
    symbols: list[str],
    horizon_steps: int = 5,
    up_pct: float = 0.002,
    down_pct: float = 0.002,
    adaptive_thresholds: bool = False,
    threshold_k: float = 2.5,
    threshold_vol_window: int = 60,
    threshold_floor: float = 0.003,
    threshold_cap: float = 0.05,
    horizon_steps_list: list[int] | None = None,
    min_warmup_bars: int = 0,
) -> dict[str, int]:
    horizons = _validate_horizons(horizon_steps=horizon_steps, horizon_steps_list=horizon_steps_list)

    if not math.isfinite(up_pct) or not math.isfinite(down_pct):
        raise ValueError("up_pct and down_pct must be finite numbers")
    if up_pct <= 0 or down_pct <= 0:
        raise ValueError("up_pct and down_pct must be > 0")
    if adaptive_thresholds:
        if threshold_k <= 0:
            raise ValueError("threshold_k must be > 0 when adaptive_thresholds=True")
        if threshold_vol_window <= 1:
            raise ValueError("threshold_vol_window must be > 1 when adaptive_thresholds=True")
        if threshold_floor <= 0 or threshold_cap <= 0 or threshold_floor > threshold_cap:
            raise ValueError("threshold_floor/cap must be > 0 and floor <= cap")

    run_id = new_run_id(prefix="dataset")
    out_rows: list[dict[str, Any]] = []

    for symbol in symbols:
        rows: list[dict[str, Any]] = []
        for feat_file in sorted((features_root / "features" / symbol).glob("part-*.jsonl")):
            rows.extend(_read_jsonl(feat_file))
        if not rows:
            continue
        rows.sort(key=lambda r: int(r.get("ts", 0) or 0))

        prices = [float(r.get("mid_price", 0.0)) for r in rows]
        ret_1_vals = [float(r.get("ret_1", 0.0)) for r in rows]
        # Exclude synthetic gap-fill rows and warm-up rows whose long-lookback
        # features are still degenerate (Q6): require enough observed history.
        clean_idx = [
            i
            for i, r in enumerate(rows)
            if not bool(r.get("is_synthetic", False)) and int(r.get("obs", 0) or 0) >= min_warmup_bars
        ]

        for i in clean_idx:
            row_up = up_pct
            row_down = down_pct
            if adaptive_thresholds:
                thr = _adaptive_threshold(
                    ret_1=ret_1_vals,
                    idx=i,
                    k=threshold_k,
                    window=threshold_vol_window,
                    floor=threshold_floor,
                    cap=threshold_cap,
                )
                row_up = thr
                row_down = thr

            for horizon in horizons:
                # Skip right-censored rows: without a full forward window the
                # outcome is unknown, and labeling it "no event" biases the base
                # rate downward.
                if i + horizon >= len(prices):
                    continue
                future = prices[i : i + horizon + 1]
                labels = label_event_path(future, up_pct=row_up, down_pct=row_down)
                out_row = {name: rows[i].get(name, 0.0) for name in MODEL_FEATURES}
                out_row.update(
                    {
                        "symbol": symbol,
                        "ts": rows[i]["ts"],
                        "mid_price": rows[i].get("mid_price", 0.0),
                        "obs": rows[i].get("obs", 0),
                        "up_hit": labels["up_hit"],
                        "down_hit": labels["down_hit"],
                        "first_hit": labels["first_hit"],
                        "tte_idx": labels["tte_idx"],
                        "horizon_steps": horizon,
                        "up_pct": row_up,
                        "down_pct": row_down,
                    }
                )
                out_rows.append(out_row)

    # Global chronological order so downstream walk-forward splits slice time,
    # not symbol blocks.
    out_rows.sort(key=lambda r: (int(r["ts"]), str(r["symbol"]), int(r["horizon_steps"])))

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
            "horizons": horizons,
            "up_pct": up_pct,
            "down_pct": down_pct,
            "adaptive_thresholds": adaptive_thresholds,
            "threshold_k": threshold_k,
            "threshold_vol_window": threshold_vol_window,
            "threshold_floor": threshold_floor,
            "threshold_cap": threshold_cap,
        },
    )
    write_manifest(dataset_dir / "manifest.json", manifest)
    return {"rows": len(out_rows), "run_id": run_id}
