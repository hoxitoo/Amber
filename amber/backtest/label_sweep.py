"""Read-only sweep over label definitions: horizon x barrier ruler x label shape.

The question this answers is not "which settings are most profitable" but
"which definition of a pump can this feature set actually predict, and does the
edge survive the delay between the signal and acting on it".

Two findings drove the design:

- Precision at the operating point measured 36.8% on the scoring bar (lift x1.80
  over a 20.4% base rate) but ~18.8% once entry was delayed by one bar — roughly
  the base rate. If the edge dies inside a minute, no barrier setting rescues it,
  so `entry_lag_bars` is a first-class axis here rather than a fixed assumption.

- `range_atr_14` carries ~32% of permutation importance, i.e. the model largely
  forecasts volatility. The production barrier is `k * sigma_fast * sqrt(h)`
  scaled by the *same* fast volatility, so a correctly detected volatility spike
  widens the target proportionally and cancels itself. The `slow_vol` ruler keeps
  cross-symbol comparability (a fixed barrier means something different on BTC
  than on a small cap) while removing that coupling.

Nothing here writes datasets or models. Arms are trained in memory and the
result is a JSON report, so the sweep can run against a live box without
disturbing the labels the running system was built on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

from amber.labeling.events import label_event_path
from amber.models.features import MODEL_FEATURES
from amber.models.split import make_holdout_splits

logger = logging.getLogger(__name__)

RULERS = ("fast_vol", "slow_vol", "fixed_070", "fixed_100")
SHAPES = ("two_sided", "one_sided")
SLOW_VOL_WINDOW = 1440  # 24h of 1m candles: a baseline the fast signal moves against

# Alert budget: precision is compared at a FIXED number of alerts rather than at
# each arm's own tuned threshold. Tuning a threshold per arm would let an arm win
# by firing rarely on easy rows, and would fold calibration quality into a
# measurement meant to be about the label.
#
# One arm scores one row per symbol per candle, so alerts/day = budget x symbols
# x 1440. At 27 symbols, 0.002 is ~78 alerts/day (~3/hour) — a rate a human can
# actually look at. Raise it to compare arms at higher recall.
DEFAULT_BUDGET = 0.002


@dataclass
class _Series:
    """One symbol's feature rows plus the price/return arrays labels need."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    prices: list[float] = field(default_factory=list)
    ret_1: list[float] = field(default_factory=list)
    clean_idx: list[int] = field(default_factory=list)


def _rolling_vol(values: Sequence[float], end_idx: int, window: int) -> float:
    start = max(0, end_idx - window + 1)
    chunk = values[start : end_idx + 1]
    if not chunk:
        return 0.0
    mean = sum(chunk) / len(chunk)
    var = sum((x - mean) ** 2 for x in chunk) / len(chunk)
    return math.sqrt(max(0.0, var))


def label_one_sided(prices: Sequence[float], up_pct: float, down_pct: float) -> dict[str, int]:
    """Did price reach +up_pct / -down_pct at any point in the window.

    The production label is a triple barrier, which encodes a stop-loss: a run
    that dips first and then rallies counts as a loss. That matches a mechanical
    trade, but Amber's output is an alert claiming "a pump is coming", and this
    is the label that claim actually corresponds to. It is also what
    `quality_report._confirmed_outcome` already uses to score live signals, so
    the offline and online definitions agree under this shape.
    """
    if len(prices) < 2:
        return {"up_hit": 0, "down_hit": 0}
    p0 = prices[0]
    up_level = p0 * (1.0 + up_pct)
    down_level = p0 * (1.0 - down_pct)
    up = any(p >= up_level for p in prices[1:])
    down = any(p <= down_level for p in prices[1:])
    return {"up_hit": int(up), "down_hit": int(down)}


def load_series(
    features_root: Path,
    *,
    max_candles_per_symbol: int = 2880,
    min_warmup_bars: int = 60,
    vol_context: int = SLOW_VOL_WINDOW,
) -> dict[str, _Series]:
    """Feature rows per symbol, windowed exactly like the production dataset.

    An extra `vol_context` prefix is kept ahead of the labelled window so the
    slow-volatility ruler has a full lookback at the first labelled row; those
    prefix rows are never labelled themselves.
    """
    from amber.common.jsonl import read_tail

    out: dict[str, _Series] = {}
    features_dir = features_root / "features"
    if not features_dir.exists():
        return out

    for symbol_dir in sorted(p for p in features_dir.iterdir() if p.is_dir()):
        want = max_candles_per_symbol + vol_context if max_candles_per_symbol > 0 else 0
        rows: list[dict[str, Any]] = []
        parts = sorted(symbol_dir.glob("part-*.jsonl"))
        for part in reversed(parts):
            rows = (read_tail(part, want) if want else read_tail(part, 10**9)) + rows
            if want and len(rows) >= want:
                break
        if want:
            rows = rows[-want:]
        if len(rows) < min_warmup_bars + 2:
            continue

        rows.sort(key=lambda r: int(r.get("ts", 0) or 0))
        label_start = max(0, len(rows) - max_candles_per_symbol) if max_candles_per_symbol > 0 else 0
        series = _Series(
            rows=rows,
            prices=[float(r.get("mid_price", 0.0) or 0.0) for r in rows],
            ret_1=[float(r.get("ret_1", 0.0) or 0.0) for r in rows],
        )
        series.clean_idx = [
            i
            for i, r in enumerate(rows)
            if i >= label_start
            and not bool(r.get("is_synthetic", False))
            and int(r.get("obs", 0) or 0) >= min_warmup_bars
        ]
        if series.clean_idx:
            out[symbol_dir.name] = series
    return out


def _barrier(
    series: _Series, i: int, horizon: int, ruler: str, k: float, floor: float, cap: float
) -> tuple[float, str]:
    """Half-width of the barrier for row `i`, plus whether a clamp decided it.

    Volatility windows end at i-1 in every case: the current bar's `ret_1` is a
    model feature, so letting it also set the target couples label to input.

    The clamp flag matters more than it looks. `threshold_floor` is 0.5%, and at
    1m crypto volatility `k * sigma * sqrt(15)` often lands below that — when it
    does, a "volatility-scaled" barrier is really a fixed one, and comparing
    rulers measures nothing. A high floored share is therefore a finding about
    the current configuration, not a detail of this sweep.
    """
    if ruler == "fixed_070":
        return 0.007, "none"
    if ruler == "fixed_100":
        return 0.010, "none"
    window = SLOW_VOL_WINDOW if ruler == "slow_vol" else 60
    vol = _rolling_vol(series.ret_1, end_idx=max(0, i - 1), window=window)
    raw = k * vol * math.sqrt(max(1, horizon))
    if raw < floor:
        return floor, "floor"
    if raw > cap:
        return cap, "cap"
    return raw, "none"


def build_arm_rows(
    series_by_symbol: dict[str, _Series],
    *,
    horizon: int,
    ruler: str,
    shape: str,
    k: float = 0.8,
    floor: float = 0.005,
    cap: float = 0.05,
) -> list[dict[str, Any]]:
    """Label every clean row under one arm. Censored rows are dropped, not
    labelled negative, which would bias the base rate downward."""
    labeller = label_one_sided if shape == "one_sided" else label_event_path
    out: list[dict[str, Any]] = []
    for symbol, series in series_by_symbol.items():
        prices = series.prices
        n = len(prices)
        for i in series.clean_idx:
            if i + horizon >= n:
                continue
            half, clamp = _barrier(series, i, horizon, ruler, k, floor, cap)
            labels = labeller(prices[i : i + horizon + 1], up_pct=half, down_pct=half)
            src = series.rows[i]
            row = {name: src.get(name, 0.0) for name in MODEL_FEATURES}
            row.update(
                {
                    "symbol": symbol,
                    "ts": int(src.get("ts", 0) or 0),
                    "up_hit": int(labels["up_hit"]),
                    "down_hit": int(labels["down_hit"]),
                    "horizon_steps": horizon,
                    "up_pct": half,
                    "down_pct": half,
                    "_i": i,
                    "_clamp": clamp,
                }
            )
            out.append(row)
    out.sort(key=lambda r: (r["ts"], r["symbol"]))
    return out


def family_z(n_arms: int, alpha: float = 0.05) -> float:
    """One-sided z for a family-wise error rate of `alpha` across `n_arms`.

    Ranking 24 arms and reporting the best is 24 chances to cross a 95% bound,
    and something usually does. Measured: on a 27-symbol pure random walk with
    no edge at all, per-arm 95% bounds returned a top arm at lift 1.00 and the
    verdict `edge_survives_lag` — a false positive on the null, from the exact
    tool built to keep the project off noise. Bonferroni splits alpha across the
    family, so the bound answers "better than random after looking 24 times".
    """
    from statistics import NormalDist

    return NormalDist().inv_cdf(1.0 - alpha / max(1, n_arms))


def _wilson_low(hits: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson interval for a proportion at confidence `z`.

    The test segment of a 48h window is a few hours, so an arm fires only tens
    of alerts and its precision carries a wide interval. Ranking arms on the
    point estimate alone would be ranking noise — which is the failure this
    whole sweep exists to avoid — so every precision is reported with the
    bound, and the verdict uses the bound, not the estimate.
    """
    if n <= 0:
        return 0.0
    p = hits / n
    d = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n)))
    return max(0.0, (centre - margin) / d)


def _episodes(timestamps: list[int], horizon: int, step_ms: int = 60_000) -> int:
    """Independent market episodes among a set of alerts.

    Alerts are not independent observations, in two ways that both inflate the
    apparent sample size:

    - Overlap. Two alerts a few bars apart look forward at overlapping windows,
      so their outcomes are largely the same event seen twice.
    - Cross-section. Crypto moves together. When the market lurches, the model
      fires across many symbols at once and every one of those alerts is the
      same underlying event.

    Grouping purely by time handles both: alerts within one horizon of each
    other, on any symbol, count once. This is the effective sample size the
    interval should use, and it can be an order of magnitude below the alert
    count.
    """
    if not timestamps:
        return 0
    ordered = sorted(timestamps)
    span = max(1, horizon) * step_ms
    count = 1
    anchor = ordered[0]
    for ts in ordered[1:]:
        if ts - anchor > span:
            count += 1
            anchor = ts
    return count


def _precision_at_budget(
    scores: list[float],
    labels: list[int],
    timestamps: list[int],
    budget: float,
    z: float = 1.96,
    horizon: int = 15,
) -> dict[str, Any]:
    """Precision over the highest-scoring `budget` fraction of rows.

    A fixed alert budget, rather than a per-arm threshold, is what makes arms
    comparable: otherwise an arm can post high precision purely by firing less.
    """
    n = len(scores)
    if n == 0:
        return {"precision": None, "base_rate": None, "lift": None, "alerts": 0}
    take = max(1, int(round(n * budget)))
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:take]
    hits = sum(labels[i] for i in order)
    base = sum(labels) / n
    precision = hits / take
    low = _wilson_low(hits, take, z)

    # The same bound recomputed on episodes rather than alerts. This is the one
    # the verdict uses: 117 alerts drawn from 5 market lurches carry about as
    # much evidence as 5 observations, and the naive interval would be wrong by
    # a factor of five.
    eps = _episodes([timestamps[i] for i in order], horizon)
    eps_hits = int(round(precision * eps))
    low_clustered = _wilson_low(eps_hits, eps, z) if eps > 0 else 0.0
    return {
        "precision": precision,
        "precision_ci_low": low,
        "base_rate": base,
        "lift": (precision / base) if base > 0 else None,
        # Lift the data supports at 95%, not the lift that happened to land.
        "lift_ci_low": (low / base) if base > 0 else None,
        "episodes": eps,
        "lift_ci_low_clustered": (low_clustered / base) if base > 0 else None,
        "alerts": take,
        "hits": hits,
        "rows": n,
    }


def _lagged(
    rows: list[dict[str, Any]], scores: list[float], lag: int, label_key: str
) -> tuple[list[float], list[int], list[int]]:
    """Pair each row's score with the label `lag` bars later on the same symbol.

    lag=0 scores and grades the same bar. lag=1 is what a human acting on an
    alert actually gets: the alert is emitted once the bar closes, so the
    outcome that matters starts from the next bar.
    """
    if lag <= 0:
        return scores, [int(r[label_key]) for r in rows], [int(r["ts"]) for r in rows]
    by_key = {(r["symbol"], r["_i"]): int(r[label_key]) for r in rows}
    out_s: list[float] = []
    out_y: list[int] = []
    out_ts: list[int] = []
    for score, row in zip(scores, rows):
        target = by_key.get((row["symbol"], row["_i"] + lag))
        if target is None:
            continue  # no labelled row that far forward; dropping beats guessing
        out_s.append(score)
        out_y.append(target)
        out_ts.append(int(row["ts"]))
    return out_s, out_y, out_ts


def evaluate_arm(
    rows: list[dict[str, Any]],
    *,
    budget: float = DEFAULT_BUDGET,
    lags: Sequence[int] = (0, 1),
    train_frac: float = 0.7,
    calib_frac: float = 0.15,
    gap: int | None = None,
    target: str = "pump",
    z: float = 1.96,
    horizon: int = 15,
) -> dict[str, Any]:
    """Train on the train segment, report precision on calib and test.

    The purge gap defaults to the horizon: a fixed 30 would let a 60-bar label
    near the train boundary see 30 bars of the test segment.
    """
    gap = max(30, horizon) if gap is None else gap
    from amber.models.dataset_io import order_with_pseudo_time, split_rows
    from amber.models.train import _fit_dual, _predict_head

    label_key = "up_hit" if target == "pump" else "down_hit"
    ordered, pts, _mode = order_with_pseudo_time(rows)
    splits = make_holdout_splits(pts, train_frac=train_frac, calib_frac=calib_frac, gap=gap)
    if splits is None:
        return {"status": "too_small", "rows": len(rows)}
    seg = split_rows(ordered, pts, splits)
    if not seg["train"] or not seg["test"]:
        return {"status": "empty_segment", "rows": len(rows)}
    if len({int(r[label_key]) for r in seg["train"]}) < 2:
        return {"status": "single_class_train", "rows": len(rows)}

    model = _fit_dual(seg["train"])
    if model["model_type"] == "constant_dual_v1":
        return {"status": "degenerate_model", "rows": len(rows)}

    span_ms = max(1, pts[-1] - pts[0])
    days = span_ms / 86_400_000.0
    out: dict[str, Any] = {
        "status": "ok",
        "model_type": model["model_type"],
        "rows": len(rows),
        "train_rows": len(seg["train"]),
        "span_days": days,
    }
    for name in ("calib", "test"):
        part = seg[name]
        if not part:
            out[name] = {"status": "empty"}
            continue
        raw = _predict_head(model, part, target=target)
        per_lag: dict[str, Any] = {}
        for lag in lags:
            s, y, ts = _lagged(part, raw, lag, label_key)
            res = _precision_at_budget(s, y, ts, budget, z, horizon)
            # Alerts per day at this budget, so an arm's precision can be read
            # against how much attention it costs.
            frac = len(part) / len(ordered) if ordered else 0.0
            res["alerts_per_day"] = (res["alerts"] / (days * frac)) if days > 0 and frac > 0 else None
            per_lag[f"lag{lag}"] = res
        out[name] = per_lag
    return out


def run_sweep(
    features_root: Path,
    *,
    horizons: Sequence[int] = (15, 30, 60),
    rulers: Sequence[str] = RULERS,
    shapes: Sequence[str] = SHAPES,
    budget: float = DEFAULT_BUDGET,
    max_candles_per_symbol: int = 2880,
    min_warmup_bars: int = 60,
    k: float = 0.8,
    floor: float = 0.005,
    cap: float = 0.05,
    target: str = "pump",
) -> dict[str, Any]:
    series = load_series(
        features_root,
        max_candles_per_symbol=max_candles_per_symbol,
        min_warmup_bars=min_warmup_bars,
    )
    if not series:
        return {"status": "no_features", "arms": []}

    # Every arm is scored against a bound corrected for how many arms are being
    # compared, so "the best of 24" has to clear a higher bar than "the only one".
    n_arms = max(1, len(horizons) * len(rulers) * len(shapes))
    z = family_z(n_arms)

    arms: list[dict[str, Any]] = []
    for horizon in horizons:
        for ruler in rulers:
            for shape in shapes:
                rows = build_arm_rows(
                    series, horizon=horizon, ruler=ruler, shape=shape, k=k, floor=floor, cap=cap
                )
                res = evaluate_arm(rows, budget=budget, target=target, z=z, horizon=horizon)
                res.update({"horizon": horizon, "ruler": ruler, "shape": shape})
                # Average realised barrier: a ruler is only interpretable next to
                # the move size it actually asks for. `floored_pct` says how
                # often the configured floor, not the ruler, set the target.
                if rows:
                    res["avg_barrier_pct"] = sum(r["up_pct"] for r in rows) / len(rows) * 100.0
                    res["floored_pct"] = sum(r["_clamp"] == "floor" for r in rows) / len(rows) * 100.0
                    res["capped_pct"] = sum(r["_clamp"] == "cap" for r in rows) / len(rows) * 100.0
                arms.append(res)
                logger.info(
                    "arm h=%s ruler=%s shape=%s -> %s",
                    horizon, ruler, shape, res.get("status"),
                )
                del rows  # one arm at a time: the box has hit OOM before

    ok = [a for a in arms if a.get("status") == "ok"]

    # Rank by the lift the data SUPPORTS at 95%, not the lift that landed. On a
    # pure random walk with no edge at all, a 20-alert test segment produced a
    # top arm at lift 1.78 — ranking on the point estimate would have sent us
    # off to redefine the target on the strength of noise.
    def _lag1(a: dict[str, Any], segment: str) -> dict[str, Any]:
        return (a.get(segment) or {}).get("lag1") or {}

    def _test_lag1_lift_low(a: dict[str, Any]) -> float:
        # Clustered bound: alerts from one market lurch are one observation.
        return _lag1(a, "test").get("lift_ci_low_clustered") or 0.0

    ranked = sorted(ok, key=_test_lag1_lift_low, reverse=True)
    best = ranked[0] if ranked else None
    verdict = "no_edge"
    if best is not None:
        # Confirmed twice: significant on test, and reproduced on the calib
        # segment it was not selected against.
        if _test_lag1_lift_low(best) > 1.0 and (_lag1(best, "calib").get("lift") or 0.0) > 1.0:
            verdict = "edge_survives_lag"
        elif _test_lag1_lift_low(best) > 1.0:
            verdict = "test_only"  # not reproduced on calib: treat as noise
        elif (_lag1(best, "test").get("lift") or 0.0) > 1.15:
            # Promising point estimate the interval does not support. Named
            # apart from the `underpowered` flag below, which is about how many
            # alerts were fired rather than what they showed.
            verdict = "inconclusive"

    alerts = (_lag1(best, "test").get("alerts") or 0) if best else 0
    return {
        "status": "ok",
        "symbols": len(series),
        "budget": budget,
        "target": target,
        "arms": arms,
        "best": best,
        "verdict": verdict,
        "n_arms": n_arms,
        "family_z": z,
        "test_alerts_per_arm": alerts,
        # Too few alerts and every arm's interval swallows the differences. Say
        # so in the report rather than letting the table imply a clean ranking.
        "underpowered": alerts < 60,
        "baseline_arm": next(
            (a for a in ok if a["horizon"] == 15 and a["ruler"] == "fast_vol" and a["shape"] == "two_sided"),
            None,
        ),
    }


def format_table(report: dict[str, Any]) -> str:
    """Human-readable table, ordered the way the decision is made."""
    if report.get("status") != "ok":
        return f"sweep unavailable: {report.get('status')}"
    header = (
        f"{'h':>3} {'ruler':<10} {'shape':<10} {'barrier%':>8} {'floored%':>8} "
        f"{'base':>6} {'P@lag0':>7} {'lift0':>6} {'P@lag1':>7} {'lift1':>6} {'lift1_lo':>8} "
        f"{'episodes':>8} {'lift1_ep':>8} {'alerts/d':>9}"
    )
    lines = [header, "-" * len(header)]
    ok = [a for a in report["arms"] if a.get("status") == "ok"]
    for a in sorted(
        ok,
        key=lambda x: -(((x.get("test") or {}).get("lag1") or {}).get("lift_ci_low_clustered") or 0.0),
    ):
        t = a.get("test") or {}
        l0, l1 = t.get("lag0") or {}, t.get("lag1") or {}

        def _f(v: Any, spec: str = "{:.3f}") -> str:
            return spec.format(v) if isinstance(v, (int, float)) else "—"

        lines.append(
            f"{a['horizon']:>3} {a['ruler']:<10} {a['shape']:<10} "
            f"{_f(a.get('avg_barrier_pct'), '{:.3f}'):>8} "
            f"{_f(a.get('floored_pct'), '{:.0f}'):>8} "
            f"{_f(l1.get('base_rate')):>6} {_f(l0.get('precision')):>7} {_f(l0.get('lift'), '{:.2f}'):>6} "
            f"{_f(l1.get('precision')):>7} {_f(l1.get('lift'), '{:.2f}'):>6} "
            f"{_f(l1.get('lift_ci_low'), '{:.2f}'):>8} "
            f"{_f(l1.get('episodes'), '{:.0f}'):>8} "
            f"{_f(l1.get('lift_ci_low_clustered'), '{:.2f}'):>8} "
            f"{_f(l1.get('alerts_per_day'), '{:.1f}'):>9}"
        )
    skipped = [a for a in report["arms"] if a.get("status") != "ok"]
    if skipped:
        lines.append("")
        for a in skipped:
            lines.append(f"  skipped h={a['horizon']} {a['ruler']}/{a['shape']}: {a.get('status')}")
    lines.append("")
    lines.append(f"verdict: {report.get('verdict')}")
    lines.append(
        f"ranked by lift1_ep, not lift1: the lift supported after comparing {report.get('n_arms')} arms "
        f"(family-wise 95%, z={report.get('family_z', 0.0):.2f}) AND after collapsing alerts that "
        "belong to the same market episode into one observation."
    )
    lines.append(
        "lift1_lo treats every alert as independent, which it is not: alerts within one horizon "
        "of each other, on any symbol, are one event. Read lift1_ep. <= 1.00 means the arm is "
        "indistinguishable from firing at random."
    )
    if report.get("underpowered"):
        lines.append(
            f"WARNING: only {report.get('test_alerts_per_arm')} alerts per arm on the test segment. "
            "Every interval is wide enough to swallow the differences between arms — "
            "raise --budget or --max-candles before drawing conclusions."
        )
    return "\n".join(lines)


def save_report(logs_root: Path, report: dict[str, Any]) -> Path:
    logs_root.mkdir(parents=True, exist_ok=True)
    out = logs_root / "label_sweep.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
