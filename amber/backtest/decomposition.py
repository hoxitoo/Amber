"""Is the edge about magnitude or about direction? (roadmap D1)

The project's goal names two questions — catch volatility early, and determine
direction — and the production label merges them. `up_hit` means "rose 1%", so

    P(up_hit) = P(move) x P(up | move)

and lift on `up_hit` can come from either factor. A model that forecasts
magnitude perfectly and knows nothing at all about sign still scores lift ~2 on
that label, because roughly half of large moves are up. Since `range_atr_14`
carries 32.4% of permutation importance, the model is substantially a
volatility forecaster and the split is not hypothetical.

This module trains one head per factor and measures each separately:

- `move`      — will |move| reach the barrier, either way. The volatility
                question, over all rows.
- `direction` — given the barrier was reached, was it the upper one. The sign
                question, over the move subset ONLY. Measured on all rows it
                would be contaminated by magnitude, which is the whole mistake
                being corrected here.
- `pump`      — the production target, kept for reference so the decomposition
                can be checked against it.

`direction` has a base rate near 0.5 by construction, so its lift is read
against a coin flip: lift ~1.0 means no directional skill, and the entire pump
lift was magnitude.

Read-only. Trains in memory, writes only the report the caller saves.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from amber.backtest.label_sweep import (
    _episodes,
    _precision_at_budget,
    build_arm_rows,
    family_z,
    load_series,
)
from amber.models.features import MODEL_FEATURES, feature_vector
from amber.models.split import make_holdout_splits

logger = logging.getLogger(__name__)

TARGETS = ("pump", "move", "direction")


def add_factor_labels(rows: list[dict[str, Any]]) -> int:
    """Attach `move_hit` and `dir_up` in place; return the inconsistent count.

    `dir_up` is only defined where a barrier was actually reached; elsewhere it
    is None so the caller cannot silently score a row that has no direction.

    A row that touched a barrier but carries no `first_hit` is contradictory,
    and it is returned as a count rather than tolerated quietly. That exact
    combination was a live bug: `build_arm_rows` did not copy `first_hit`
    through, so every `dir_up` came out None while `move_hit` still worked off
    the flags — the direction head silently had nothing to train on, and only
    the empty test segment gave it away.
    """
    inconsistent = 0
    for row in rows:
        first = int(row.get("first_hit", 0) or 0)
        flags = int(row.get("up_hit", 0)) or int(row.get("down_hit", 0))
        if flags and first == 0:
            inconsistent += 1
        row["move_hit"] = int(bool(first != 0 or flags))
        row["dir_up"] = int(first == 1) if first != 0 else None
    return inconsistent


def _matrix(rows: list[dict[str, Any]]) -> list[list[float]]:
    return [feature_vector(r) for r in rows]


def _fit(rows: list[dict[str, Any]], label_key: str) -> dict[str, Any] | None:
    """One head on `label_key`, in the same shape `_predict_matrix` expects."""
    from amber.models.train import _apply_clip, _clip_bounds, _fit_head

    y = [int(r[label_key]) for r in rows]
    if len(set(y)) < 2:
        return None
    x = _matrix(rows)
    bounds = _clip_bounds(x)
    x = _apply_clip(x, bounds, inplace=True)
    w = [1.0 / max(1, int(r.get("horizon_steps", 1) or 1)) for r in rows]
    head = _fit_head(x, y, w)
    if head.get("type") == "constant":
        return None
    return {"features": list(MODEL_FEATURES), "heads": {"head": head}, "clip_bounds": bounds}


def _score(model: dict[str, Any], rows: list[dict[str, Any]]) -> list[float]:
    from amber.models.importance import _clipped_matrix, _predict_matrix

    matrix, _names = _clipped_matrix(model, rows)
    return _predict_matrix(model, "head", matrix)


def _measure(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    label_key: str,
    *,
    budget: float,
    z: float,
    horizon: int,
) -> dict[str, Any]:
    if not train_rows or not test_rows:
        return {"status": "empty_segment"}
    model = _fit(train_rows, label_key)
    if model is None:
        return {"status": "single_class_or_degenerate", "train_rows": len(train_rows)}

    scores = _score(model, test_rows)
    labels = [int(r[label_key]) for r in test_rows]
    ts = [int(r["ts"]) for r in test_rows]
    res = _precision_at_budget(scores, labels, ts, budget, z, horizon)
    res.update({
        "status": "ok",
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "model_type": model["heads"]["head"].get("type"),
    })
    return res


def decompose(
    features_root: Path,
    *,
    horizon: int = 15,
    barrier: float = 0.010,
    budget: float = 0.01,
    max_candles_per_symbol: int = 2880,
    min_warmup_bars: int = 60,
    train_frac: float = 0.7,
    calib_frac: float = 0.15,
) -> dict[str, Any]:
    """Lift on magnitude and on direction, measured separately."""
    from amber.models.dataset_io import order_with_pseudo_time, split_rows

    series = load_series(
        features_root,
        max_candles_per_symbol=max_candles_per_symbol,
        min_warmup_bars=min_warmup_bars,
    )
    if not series:
        return {"status": "no_features"}

    ruler = "fixed_100" if abs(barrier - 0.010) < 1e-9 else "fixed_070"
    rows = build_arm_rows(series, horizon=horizon, ruler=ruler, shape="one_sided")
    if not rows:
        return {"status": "no_rows"}
    inconsistent = add_factor_labels(rows)
    if inconsistent:
        logger.warning(
            "%s of %s rows report a barrier touch with no first_hit; direction cannot be read "
            "from them and they are excluded from the direction head",
            inconsistent, len(rows),
        )

    ordered, pts, _mode = order_with_pseudo_time(rows)
    splits = make_holdout_splits(pts, train_frac=train_frac, calib_frac=calib_frac, gap=max(30, horizon))
    if splits is None:
        return {"status": "too_small", "rows": len(rows)}
    seg = split_rows(ordered, pts, splits)

    # Three heads compared, so the bound is corrected for three looks.
    z = family_z(len(TARGETS))
    out: dict[str, Any] = {
        "status": "ok",
        "symbols": len(series),
        "horizon": horizon,
        "barrier_pct": barrier * 100.0,
        "budget": budget,
        "rows": len(rows),
        "family_z": z,
        "rows_missing_first_hit": inconsistent,
    }

    for target, label_key in (("pump", "up_hit"), ("move", "move_hit")):
        out[target] = _measure(
            seg["train"], seg["test"], label_key, budget=budget, z=z, horizon=horizon
        )

    # Direction lives on the move subset only. Scoring it over all rows would
    # let the magnitude signal leak straight back in, which is the confusion
    # this whole module exists to undo.
    tr_moves = [r for r in seg["train"] if r["dir_up"] is not None]
    te_moves = [r for r in seg["test"] if r["dir_up"] is not None]

    # The move subset is a fraction of all rows, so the same budget fraction
    # would fire a fraction of the alerts and the direction head would look
    # weaker purely for lack of observations. Scale the budget to fire the same
    # ABSOLUTE number of alerts, so the two lifts are compared at equal power.
    dir_budget = budget
    if te_moves and seg["test"]:
        dir_budget = min(1.0, budget * len(seg["test"]) / len(te_moves))

    out["direction"] = _measure(
        tr_moves, te_moves, "dir_up", budget=dir_budget, z=z, horizon=horizon
    )
    out["direction"]["move_rows_test"] = len(te_moves)
    out["direction"]["budget"] = dir_budget
    out["direction"]["episodes_available"] = _episodes([int(r["ts"]) for r in te_moves], horizon)

    out["verdict"] = _verdict(out)
    return out


def _verdict(report: dict[str, Any]) -> str:
    """What the two lifts mean for the product, not just for the metric."""
    d = report.get("direction") or {}
    m = report.get("move") or {}
    if d.get("status") != "ok" or m.get("status") != "ok":
        return "incomplete"

    d_low = d.get("lift_ci_low_clustered") or 0.0
    m_low = m.get("lift_ci_low_clustered") or 0.0
    if d_low > 1.0 and m_low > 1.0:
        return "both"  # volatility AND direction: the goal as stated
    if m_low > 1.0:
        return "magnitude_only"  # a volatility scanner; direction is a coin flip
    if d_low > 1.0:
        return "direction_only"
    return "neither"


def format_report(report: dict[str, Any]) -> str:
    if report.get("status") != "ok":
        return f"decomposition unavailable: {report.get('status')}"

    def _f(v: Any, spec: str = "{:.3f}") -> str:
        return spec.format(v) if isinstance(v, (int, float)) else "—"

    lines = [
        f"horizon {report['horizon']} bars · barrier {report['barrier_pct']:.2f}% · "
        f"{report['symbols']} symbols · {report['rows']:,} rows · "
        f"family-wise 95% over {len(TARGETS)} targets (z={report['family_z']:.2f})",
        "",
        f"{'target':<11} {'question':<34} {'base':>6} {'precision':>10} {'lift':>7} {'lift_lo':>8} {'episodes':>9}",
        "-" * 92,
    ]
    questions = {
        "pump": "will it rise 1% (production)",
        "move": "will it move 1% either way",
        "direction": "given a move, was it up",
    }
    for target in TARGETS:
        r = report.get(target) or {}
        if r.get("status") != "ok":
            lines.append(f"{target:<11} {questions[target]:<34} {r.get('status', '—')}")
            continue
        lines.append(
            f"{target:<11} {questions[target]:<34} "
            f"{_f(r.get('base_rate')):>6} {_f(r.get('precision')):>10} "
            f"{_f(r.get('lift'), '{:.2f}'):>7} {_f(r.get('lift_ci_low_clustered'), '{:.2f}'):>8} "
            f"{_f(r.get('episodes'), '{:.0f}'):>9}"
        )

    lines.append("")
    verdict = report.get("verdict")
    lines.append(f"verdict: {verdict}")
    lines.append({
        "both": "Volatility AND direction both carry signal — the goal as stated is reachable.",
        "magnitude_only": (
            "Volatility carries signal; direction is a coin flip. The honest product is a "
            "scanner of starting volatility, with direction left to the human reading the chart. "
            "Tick-level order flow and L2 depth (D2/D7) would be spent sharpening a signal that "
            "is not there — do not buy them on this evidence."
        ),
        "direction_only": "Direction carries signal but magnitude does not, which is unusual — re-check the labels.",
        "neither": "Neither factor is distinguishable from random at this budget and window.",
        "incomplete": "A head could not be fit; see the per-target status above.",
    }.get(verdict or "", ""))

    # The arithmetic check: P(up_hit) should be about P(move) x P(up | move).
    m, d, p = report.get("move") or {}, report.get("direction") or {}, report.get("pump") or {}
    if all(x.get("status") == "ok" for x in (m, d, p)):
        implied = (m.get("base_rate") or 0.0) * (d.get("base_rate") or 0.0)
        lines.append("")
        lines.append(
            f"consistency: P(move)={m['base_rate']:.4f} x P(up|move)={d['base_rate']:.4f} "
            f"= {implied:.4f} vs P(up_hit)={p['base_rate']:.4f}"
        )
    return "\n".join(lines)


def save_report(logs_root: Path, report: dict[str, Any]) -> Path:
    import json

    logs_root.mkdir(parents=True, exist_ok=True)
    out = logs_root / "label_decomposition.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
