"""Precision at an alert rate a person can actually act on (roadmap D4).

Everything measured so far used a 1% alert budget — **388 alerts/day across 27
symbols**. That is a firehose, not a tool, and precision there is not the
precision anyone would experience. A usable rate is 10-20/day, which is a far
more selective point on the same curve.

This sweeps the budget over the live model's out-of-sample segment, using the
real model and calibration artifacts so the numbers correspond to what the
scanner would actually emit, and reports for each rate:

- the calibrated probability cut that produces it,
- the equivalent `prob_lift_min` so it can be written straight into
  `config/thresholds.yaml`,
- precision, lift, and the episode-clustered lower bound.

The deliverable is a threshold, not a curve. Read-only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

from amber.backtest.label_sweep import _episodes, _wilson_low, family_z
from amber.labeling.events import move_label

logger = logging.getLogger(__name__)

# Rates worth asking about: the measurement budget at the top, down through
# "watch a few an hour" to "a handful a day".
DEFAULT_RATES = (400.0, 200.0, 100.0, 50.0, 20.0, 10.0, 5.0)

# Below this many independent episodes a Wilson bound swings too wildly to act
# on: a tight rate can clear 1.0 by luck and land in the live config.
MIN_EPISODES = 10

LABEL_KEYS = {"move": "move_hit", "pump": "up_hit", "dump": "down_hit"}


def _label(row: dict[str, Any], target: str) -> int:
    if target == "move":
        return move_label(row)
    return int(row.get(LABEL_KEYS[target], 0) or 0)


def lift_for_threshold(threshold: float, base_rate: float) -> float | None:
    """The `prob_lift_min` that reproduces `threshold` at this base rate.

    Inverse of the scanner's odds-space rule, so a chosen operating point can be
    written into config in the same units the gate reads.
    """
    p = min(max(float(threshold), 0.0), 1.0)
    b = min(max(float(base_rate), 0.0), 1.0)
    if b <= 0.0 or b >= 1.0 or p >= 1.0:
        return None
    return (p / (1.0 - p)) / (b / (1.0 - b))


def operating_curve(
    datasets_root: Path,
    models_root: Path,
    *,
    target: str = "move",
    rates_per_day: Sequence[float] = DEFAULT_RATES,
) -> dict[str, Any]:
    """Precision and lift at each alert rate, on the model's test segment."""
    from amber.models.dataset_io import load_latest_dataset_rows, order_with_pseudo_time, split_rows
    from amber.models.importance import _clipped_matrix, _predict_matrix
    from amber.models.infer import load_latest_model
    from amber.signals.scorer import _load_latest_calibration, calibrated_prob_for_target

    if target not in LABEL_KEYS:
        raise ValueError(f"target must be one of {sorted(LABEL_KEYS)}, got {target!r}")

    # A missing dataset or model is the normal state right after a target change
    # or a fresh deploy, not a crash: the pipeline has not rebuilt yet.
    try:
        all_rows, dataset_run = load_latest_dataset_rows(datasets_root)
    except (ValueError, FileNotFoundError, OSError) as exc:
        return {"status": "no_dataset", "detail": str(exc)}
    if not all_rows:
        return {"status": "no_dataset", "detail": "dataset is empty"}
    try:
        model = load_latest_model(models_root)
    except (ValueError, FileNotFoundError, OSError) as exc:
        return {"status": "no_model", "detail": str(exc)}
    heads = model.get("heads", {}) if isinstance(model.get("heads"), dict) else {}
    if target not in heads:
        return {"status": "no_head", "target": target, "heads": sorted(heads)}
    calibration = _load_latest_calibration(models_root)

    ordered, pts, _mode = order_with_pseudo_time(all_rows)
    splits = model.get("splits")
    rows = split_rows(ordered, pts, splits)["test"] if isinstance(splits, dict) else ordered
    in_sample = not isinstance(splits, dict)
    if not rows:
        return {"status": "empty_test_segment"}

    labels = [_label(r, target) for r in rows]
    base = sum(labels) / len(labels)
    if base <= 0:
        return {"status": "no_positive_labels", "rows": len(rows)}

    matrix, _names = _clipped_matrix(model, rows)
    raw = _predict_matrix(model, target, matrix)
    probs = [calibrated_prob_for_target(p, calibration=calibration, target=target) for p in raw]

    ts = [int(r.get("ts", 0) or 0) for r in rows]
    span_days = max(1e-9, (max(ts) - min(ts)) / 86_400_000.0)
    horizon = int(rows[0].get("horizon_steps", 15) or 15)

    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    z = family_z(len(rates_per_day))

    points: list[dict[str, Any]] = []
    for rate in rates_per_day:
        take = int(round(rate * span_days))
        if take < 1 or take > len(order):
            points.append({"alerts_per_day": rate, "status": "out_of_range", "would_need": take})
            continue
        chosen = order[:take]
        hits = sum(labels[i] for i in chosen)
        precision = hits / take
        threshold = probs[chosen[-1]]
        eps = _episodes([ts[i] for i in chosen], horizon)
        low_clustered = _wilson_low(int(round(precision * eps)), eps, z) if eps else 0.0
        points.append({
            "status": "ok",
            "alerts_per_day": rate,
            "alerts": take,
            "threshold": threshold,
            "prob_lift_min_equivalent": lift_for_threshold(threshold, base),
            "precision": precision,
            "lift": precision / base,
            "episodes": eps,
            "lift_ci_low_clustered": (low_clustered / base) if base > 0 else None,
        })

    ok = [p for p in points if p["status"] == "ok"]
    # The recommendation: the most selective rate whose edge still holds after
    # clustering AND rests on enough independent episodes to mean anything.
    #
    # The episode floor is not decoration. Without it this rule picked 10
    # alerts/day off TWO episodes while rejecting 20 alerts/day at three — a
    # non-monotonic result that is a lucky draw, not a better operating point.
    # A Wilson bound over a handful of episodes swings wildly, so a tight rate
    # can clear it by accident and would then be written into the live config.
    holding = [
        p for p in ok
        if (p.get("lift_ci_low_clustered") or 0.0) > 1.0 and (p.get("episodes") or 0) >= MIN_EPISODES
    ]
    recommended = min(holding, key=lambda p: p["alerts_per_day"]) if holding else None
    for p in ok:
        p["underpowered"] = (p.get("episodes") or 0) < MIN_EPISODES

    return {
        "status": "ok",
        "target": target,
        "dataset_run": dataset_run,
        "rows": len(rows),
        "in_sample": in_sample,
        "base_rate": base,
        "span_days": span_days,
        "horizon": horizon,
        "family_z": z,
        "points": points,
        "recommended": recommended,
    }


def format_curve(report: dict[str, Any]) -> str:
    if report.get("status") != "ok":
        status = report.get("status")
        hint = {
            "no_dataset": "the pipeline has not built a dataset yet — wait for a retrain cycle",
            "no_model": "no model has been trained yet — wait for a retrain cycle",
            "empty_test_segment": "dataset too small for a holdout split; wait for more history",
            "no_positive_labels": "no events in the test segment; wait for more history",
            "no_head": f"model has heads {report.get('heads')} — retrain to get the requested one",
        }.get(status, "")
        detail = report.get("detail", "")
        return " ".join(x for x in (f"operating curve unavailable: {status}", hint, detail) if x)

    def _f(v: Any, spec: str = "{:.3f}") -> str:
        return spec.format(v) if isinstance(v, (int, float)) else "—"

    lines = [
        f"target {report['target']} · base rate {report['base_rate']:.4f} · "
        f"{report['rows']:,} test rows over {report['span_days']:.2f} days · "
        f"family-wise 95% over {len(report['points'])} rates (z={report['family_z']:.2f})",
    ]
    if report.get("in_sample"):
        lines.append("WARNING: model has no holdout splits — these numbers are IN SAMPLE")
    lines += [
        "",
        f"{'alerts/day':>10} {'порог':>8} {'lift_min':>9} {'precision':>10} {'lift':>7} "
        f"{'lift_lo':>8} {'episodes':>9}",
        "-" * 68,
    ]
    for p in report["points"]:
        if p["status"] != "ok":
            lines.append(f"{p['alerts_per_day']:>10.0f} {'—':>8}  {p['status']} (needs {p.get('would_need')} rows)")
            continue
        lines.append(
            f"{p['alerts_per_day']:>10.0f} {_f(p['threshold'], '{:.4f}'):>8} "
            f"{_f(p['prob_lift_min_equivalent'], '{:.2f}'):>9} {_f(p['precision']):>10} "
            f"{_f(p['lift'], '{:.2f}'):>7} {_f(p['lift_ci_low_clustered'], '{:.2f}'):>8} "
            f"{_f(p['episodes'], '{:.0f}'):>9}"
            + ("  ← мало эпизодов" if p.get("underpowered") else "")
        )

    rec = report.get("recommended")
    lines.append("")
    if rec:
        lines.append(
            f"recommended: {rec['alerts_per_day']:.0f} alerts/day — precision {rec['precision']:.3f} "
            f"at lift {rec['lift']:.2f} (bound {rec['lift_ci_low_clustered']:.2f})"
        )
        lines.append(
            f"  to apply, set in config/thresholds.yaml:  prob_lift_min: "
            f"{rec['prob_lift_min_equivalent']:.2f}   # or move_prob_calibrated_min: {rec['threshold']:.4f}"
        )
        lines.append(
            "  the tightest rate whose edge still holds after collapsing alerts into market "
            "episodes; tighter rates below it are listed but not supported by the data."
        )
    else:
        lines.append(
            f"recommended: none — no rate keeps a lift bound above 1.0 on at least "
            f"{MIN_EPISODES} independent episodes. Collect more history (roadmap D3) "
            "before tightening the live threshold."
        )
    return "\n".join(lines)


def save_curve(logs_root: Path, report: dict[str, Any]) -> Path:
    logs_root.mkdir(parents=True, exist_ok=True)
    out = logs_root / "operating_curve.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
