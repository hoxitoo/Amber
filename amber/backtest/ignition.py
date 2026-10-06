"""Ignition check: can a sharp move be predicted while the market is still quiet?

Every live measurement so far says the move model fires once a move is already
under way: `range_atr_14` carries ~80% of its importance, it is matched by
one-line volatility rules (baseline check), and in the forward ledger it hits
a 1% move exactly as often as the `range_atr_14` rule does. That answers "is
price moving now?", which any volatility screener answers. The owner's
purpose is the other question — "price is calm now; will it move sharply in
the next minutes, with what probability, and why?"

This tool asks that question directly, on the data already collected. It
keeps only bars where price has been CALM — its range over the last `window`
bars is under `calm_pct` and over the last 5 bars under a fifth of that, AND
its candles' high-low range over every 20-bar slice is under `calm_pct` — so "already moving" is excluded by construction and
volatility-now cannot answer it. On those rows it labels whether price then
travels `barrier` (either way) within `horizon` bars, entering on the bar
AFTER the alert bar (an alert is only seen once its bar closes). A model is
trained on the earliest part of the window and scored on the latest, and
compared with every single feature used as a one-line rule in both
directions, at the same alert budget.

Verdicts use the episode-clustered lower bound with a family-wise correction
over everything compared (arms x (model + every rule)), so ranking 4 arms and
49 rules and keeping the best does not by itself produce a finding:

- `precursor_found`: the model's lower bound on lift clears 1 AND the best
  single rule's achieved lift — a multi-factor precursor exists.
- `signal_matched_by:<rule>`: the model's lower bound clears 1 but not the best
  rule — there is a precursor, and one feature carries it as well.
- `single_feature_signal:<rule>`: only a single feature's lower bound clears 1.
- `timing_signal:<model|rule>`: with the SAME share of alerts taken from
  every coin (within-coin ranks), the model or a feature still beats the base
  rate — "this coin is unusual for itself right now" precedes moves. This is
  the warning the owner asked for, and it outranks every verdict below.
- `coin_choice_only`: the signal is matched by a static per-coin number (the
  coin's train-mean range_atr_14): it says which coins move more, not when.
- `no_precursor`: nothing beats chance — on these inputs a quiet market gives
  no warning, and earlier warning needs earlier data (roadmap D2: order book
  and tick-level trade flow).
- `underpowered`: fewer than MIN_EPISODES independent alert episodes, or
  fewer than MIN_POSITIVE_EPISODES independent moves to predict in the test
  segment. The second condition was added after the first live run
  (2026-10-04): a 1% move from a calm market turned out to be rare (base rate
  0.1-0.5%), the 18-hour test segment of the 72h window held about ten of
  them, and the tool printed `no_precursor` when the honest reading was "too
  few events to tell". Ten events can only reveal a very strong precursor.

Because the events are rare, this check reads far more history than the
training window (`--days`, 30 by default), held as compact float32 arrays —
dict rows for 30 days x 27 symbols would not fit the box.

Read-only: trains in memory, writes only logs/ignition_check.json.
"""

from __future__ import annotations

from array import array
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from amber.backtest.label_sweep import _episodes, _precision_at_budget, _Series, family_z
from amber.labeling.events import label_path
from amber.models.features import MODEL_FEATURES
from amber.models.split import make_holdout_splits

logger = logging.getLogger(__name__)

REPORT_FILE = "ignition_check.json"
MIN_EPISODES = 10
# Independent moves (clustered like alerts) the test segment must contain
# before "no precursor" may be said. Set after the first live run, before any
# longer-history result was seen.
MIN_POSITIVE_EPISODES = 20
# Fixed before the first run. Calm = the last `window` bars moved less than
# half the target; a 1% move from there is a genuine ignition, not the
# continuation of one already in progress.
DEFAULT_WINDOWS = (30, 60)
DEFAULT_HORIZONS = (15, 30)
DEFAULT_CALM_PCT = 0.005
# ...and the last RECENT_BARS must also be still, within calm_pct / 5. The
# window test alone let the first one or two bars of a move through (a 0.2%
# bar fits inside a 0.5% range): the null fixture then "predicted" ignitions
# from ret_1, which is just the move having started. Found on that fixture,
# before any live run.
RECENT_BARS = 5
RECENT_FRACTION = 0.2
# ...and the CANDLES must be calm too, not just the per-minute mid snapshots:
# every 20-bar slice of the window must have a high-low range under calm_pct.
# Added after the 30-day live run (2026-10-04): with the mid-only filter the
# winning "precursor" was range_atr_14 — the average candle high-low — i.e.
# price whipping inside each minute while the minute snapshots stayed flat.
# That is volatility already under way, not a warning before it.
CANDLE_SLICE = 20
DEFAULT_BARRIER = 0.010
# 1% of calm test rows. Calm rows are a subset, so this is fewer alerts than
# the live scanner fires.
DEFAULT_BUDGET = 0.01
DEFAULT_DAYS = 30
_N_FEATURES = len(MODEL_FEATURES)

_VERDICT_RANK = {
    "timing_signal": 6,
    "precursor_found": 5,
    "signal_matched_by": 4,
    "single_feature_signal": 3,
    "coin_choice_only": 2,
    "no_precursor": 1.5,
    "underpowered": 1,
}


def candle_range_20(row: dict[str, Any]) -> float:
    """High-low range of the last 20 candles over the close, recovered exactly
    from the feature row: dist_to_high_20 = close/max_high - 1 and
    dist_to_low_20 = close/min_low - 1. Infinite when the row cannot say."""
    try:
        dhi = float(row.get("dist_to_high_20"))
        dlo = float(row.get("dist_to_low_20"))
    except (TypeError, ValueError):
        return float("inf")
    if 1.0 + dhi <= 0 or 1.0 + dlo <= 0:
        return float("inf")
    return max(0.0, 1.0 / (1.0 + dhi) - 1.0 / (1.0 + dlo))


def _calm_labels(
    prices: Sequence[float],
    synthetic: Sequence[bool],
    candidates: Sequence[int],
    *,
    window: int,
    horizon: int,
    calm_pct: float,
    barrier: float,
    candle_ranges: Sequence[float] | None = None,
) -> tuple[list[tuple[int, int, int, int]], int]:
    """(i, move_hit, up_hit, down_hit) for every calm candidate, plus how many
    candidates had a full window either side.

    A gap-filled bar anywhere in the lookback or the forward window disqualifies
    the row: synthetic bars are flat, so a gap would read as calm and then as
    "no move". The label starts on the bar after `i`: an alert is only seen
    once its bar has closed.
    """
    out: list[tuple[int, int, int, int]] = []
    considered = 0
    n = len(prices)
    for i in candidates:
        lo, hi = i - window + 1, i + 1 + horizon
        if lo < 0 or hi >= n:
            continue
        considered += 1
        if any(synthetic[lo : hi + 1]):
            continue
        past = prices[lo : i + 1]
        p_now = prices[i]
        if p_now <= 0 or min(past) <= 0:
            continue
        if (max(past) - min(past)) / p_now > calm_pct:
            continue  # already moving: exactly the case this check excludes
        recent = prices[max(lo, i - RECENT_BARS + 1) : i + 1]
        if (max(recent) - min(recent)) / p_now > calm_pct * RECENT_FRACTION:
            continue  # the move may have begun on the last few bars
        if candle_ranges is not None:
            # 20-bar slices ending at i, i-20, ... and one ending at the
            # window's start + 19, so together they cover every bar of it.
            ends = set(range(i, lo + CANDLE_SLICE - 2, -CANDLE_SLICE)) | {lo + CANDLE_SLICE - 1}
            if any(candle_ranges[e] >= calm_pct for e in ends if 0 <= e <= i):
                continue  # candles were swinging inside the minute: not calm
        labels = label_path(prices[i + 1 : hi + 1], up_pct=barrier, down_pct=barrier, shape="one_sided")
        up, down = int(labels["up_hit"]), int(labels["down_hit"])
        out.append((i, int(bool(up or down)), up, down))
    return out, considered


def build_ignition_rows(
    series_by_symbol: dict[str, _Series],
    *,
    window: int,
    horizon: int,
    calm_pct: float = DEFAULT_CALM_PCT,
    barrier: float = DEFAULT_BARRIER,
) -> tuple[list[dict[str, Any]], int]:
    """Dict-row view of the calm filter, for inspection and tests."""
    out: list[dict[str, Any]] = []
    considered = 0
    for symbol, series in series_by_symbol.items():
        synthetic = [bool(r.get("is_synthetic", False)) for r in series.rows]
        ranges = [candle_range_20(r) for r in series.rows]
        labelled, c = _calm_labels(series.prices, synthetic, series.clean_idx, window=window,
                                   horizon=horizon, calm_pct=calm_pct, barrier=barrier,
                                   candle_ranges=ranges)
        considered += c
        for i, move, up, down in labelled:
            src = series.rows[i]
            row = {name: src.get(name, 0.0) for name in MODEL_FEATURES}
            row.update({"symbol": symbol, "ts": int(src.get("ts", 0) or 0), "move_hit": move,
                        "up_hit": up, "down_hit": down, "horizon_steps": horizon})
            out.append(row)
    out.sort(key=lambda r: (r["ts"], r["symbol"]))
    return out, considered


class _Arm:
    """Calm rows for one (window, horizon), as compact arrays."""

    def __init__(self, window: int, horizon: int) -> None:
        self.window, self.horizon = window, horizon
        self.x = array("f")  # row-major, _N_FEATURES per row
        self.y = array("b")
        self.ts = array("q")
        self.sym = array("h")  # symbol index: the coin-choice and within-coin controls need it
        self.considered = 0


def _load_symbol(symbol_dir: Path, want: int) -> list[dict[str, Any]]:
    from amber.common.jsonl import read_tail

    rows: list[dict[str, Any]] = []
    for part in reversed(sorted(symbol_dir.glob("part-*.jsonl"))):
        rows = read_tail(part, want - len(rows)) + rows
        if len(rows) >= want:
            break
    rows.sort(key=lambda r: int(r.get("ts", 0) or 0))
    return rows


def collect_arms(
    features_root: Path,
    *,
    windows: Sequence[int],
    horizons: Sequence[int],
    calm_pct: float,
    barrier: float,
    max_candles_per_symbol: int,
    min_warmup_bars: int,
) -> tuple[list[_Arm], int]:
    """One pass over the feature files, one symbol in memory at a time."""
    arms = [_Arm(w, h) for w in windows for h in horizons]
    features_dir = features_root / "features"
    if not features_dir.exists():
        return arms, 0
    context = max(windows) + 1
    if max_candles_per_symbol <= 0:
        max_candles_per_symbol = 10**9  # all history
    symbols = 0
    for symbol_dir in sorted(p for p in features_dir.iterdir() if p.is_dir()):
        rows = _load_symbol(symbol_dir, max_candles_per_symbol + context)
        if len(rows) < min_warmup_bars + context:
            continue
        symbols += 1
        prices = [float(r.get("mid_price", 0.0) or 0.0) for r in rows]
        synthetic = [bool(r.get("is_synthetic", False)) for r in rows]
        ranges = [candle_range_20(r) for r in rows]
        start = max(0, len(rows) - max_candles_per_symbol)
        candidates = [
            i for i in range(start, len(rows))
            if not synthetic[i] and int(rows[i].get("obs", 0) or 0) >= min_warmup_bars
        ]
        for arm in arms:
            labelled, considered = _calm_labels(prices, synthetic, candidates, window=arm.window,
                                                horizon=arm.horizon, calm_pct=calm_pct, barrier=barrier,
                                                candle_ranges=ranges)
            arm.considered += considered
            for i, move, _up, _down in labelled:
                src = rows[i]
                arm.x.extend(float(src.get(name, 0.0) or 0.0) for name in MODEL_FEATURES)
                arm.y.append(move)
                arm.ts.append(int(src.get("ts", 0) or 0))
                arm.sym.append(symbols - 1)
        del rows, prices, synthetic, ranges
    return arms, symbols


def _scores(head: dict[str, Any], x: Any) -> Any:
    import numpy as np

    if head.get("type") == "lightgbm":
        import lightgbm as lgb

        return lgb.Booster(model_str=head["booster"]).predict(x)
    if head.get("type") == "logreg":
        w = np.asarray([head["weights"].get(n, 0.0) for n in MODEL_FEATURES], dtype=float)
        return 1.0 / (1.0 + np.exp(-(x @ w + float(head.get("bias", 0.0)))))
    return np.full(len(x), float(head.get("prob", 0.0)))


def _best_rule(x: Any, y: list[int], ts: list[int], *, budget: float, z: float, horizon: int) -> dict[str, Any]:
    """Every feature as a one-line rule, both directions, best achieved lift.

    Chosen on the test segment itself, which flatters the rule — deliberately:
    the model has to beat the luckiest single feature, not a fair one.
    """
    best: dict[str, Any] | None = None
    for j, name in enumerate(MODEL_FEATURES):
        col = x[:, j]
        if float(col.max()) == float(col.min()):
            continue
        for sign in (1.0, -1.0):
            res = _precision_at_budget((sign * col).tolist(), y, ts, budget, z, horizon)
            if best is None or (res.get("lift") or 0.0) > (best.get("lift") or 0.0):
                best = {**res, "rule": f"{'+' if sign > 0 else '-'}{name}"}
    return best or {"rule": None, "lift": None, "lift_ci_low_clustered": None, "episodes": 0}


def _within_coin_ranks(values: Any, sym: Any, seed: int = 11) -> Any:
    """Percentile rank of each column within its own coin, ties broken at random.

    Taking the global top q% of these picks about q% of EVERY coin's rows, so
    with no timing information the expected precision is exactly the overall
    base rate — coin choice cannot leak in. A z-score against the coin's mean
    and spread did leak it: a jumpy coin's features have heavier tails, the top
    1% of z-scores piled onto it, and pure-noise fixtures read as "timing"
    (found 2026-10-06, before any live run of this control).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    one_d = v.ndim == 1
    if one_d:
        v = v[:, None]
    out = np.empty_like(v)
    for c in np.unique(sym):
        idx = np.flatnonzero(sym == c)
        if len(idx) == 1:
            out[idx] = 0.5
            continue
        for j in range(v.shape[1]):
            order = np.lexsort((rng.random(len(idx)), v[idx, j]))
            ranks = np.empty(len(idx))
            ranks[order] = np.arange(len(idx)) / (len(idx) - 1)
            out[idx, j] = ranks
    return out[:, 0] if one_d else out


def _coin_controls(
    x_train: Any, sym_train: Any, x_test: Any, sym_test: Any, y: list[int], ts: list[int],
    model_scores: Any, *, budget: float, z: float, horizon: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Separate WHICH coin from WHEN.

    After the candle filter (2026-10-06 live run) the winning signal was still
    range_atr_14 with spread_bps second: both are mostly properties of the
    coin, not of the moment. A jumpy alt's calm candle is wider than BTC's and
    a 1% move is routine for it, so ranking by them may only pick coins.

    - coin_only: each coin scored by its mean range_atr_14 over the TRAIN
      segment — one number per coin, no timing information at all. If it does
      as well as the model, the "precursor" is coin choice.
    - within_coin: every feature ranked within its own coin (same share of
      alerts from every coin), both directions, best achieved lift.
    - model_within: the model's own scores ranked within each coin.
    The last two are "this coin is unusual for itself right now" — the warning
    the owner asked for.
    """
    import numpy as np

    j_atr = list(MODEL_FEATURES).index("range_atr_14")
    overall = float(x_train[:, j_atr].mean())
    coin_mean = {}
    for c in np.unique(sym_test):
        rows = x_train[sym_train == c, j_atr]
        coin_mean[int(c)] = float(rows.mean()) if len(rows) >= 30 else overall
    coin_score = [coin_mean[int(c)] for c in sym_test]
    coin_res = {**_precision_at_budget(coin_score, y, ts, budget, z, horizon), "rule": "coin:range_atr_14"}

    within = _best_rule(_within_coin_ranks(x_test, sym_test), y, ts, budget=budget, z=z, horizon=horizon)
    if within.get("rule"):
        within["rule"] = "rank:" + within["rule"]
    model_within = _precision_at_budget(
        _within_coin_ranks(np.asarray(model_scores, dtype=float), sym_test).tolist(), y, ts, budget, z, horizon
    )
    return coin_res, within, model_within


def _verdict(
    model: dict[str, Any],
    rule: dict[str, Any],
    positive_episodes: int,
    coin: dict[str, Any] | None = None,
    within: dict[str, Any] | None = None,
    model_within: dict[str, Any] | None = None,
) -> str:
    if (model.get("episodes") or 0) < MIN_EPISODES or positive_episodes < MIN_POSITIVE_EPISODES:
        return "underpowered"
    m_lo = model.get("lift_ci_low_clustered") or 0.0
    r_lift = rule.get("lift") or 0.0
    r_lo = rule.get("lift_ci_low_clustered") or 0.0
    within = within or {}
    coin = coin or {}
    model_within = model_within or {}
    w_lo = within.get("lift_ci_low_clustered") or 0.0
    mw_lo = model_within.get("lift_ci_low_clustered") or 0.0
    # Timing information that survives taking the same share from every coin.
    if mw_lo > 1.0 and (model_within.get("episodes") or 0) >= MIN_EPISODES and mw_lo >= w_lo:
        return "timing_signal:model"
    if w_lo > 1.0 and (within.get("episodes") or 0) >= MIN_EPISODES:
        return f"timing_signal:{within.get('rule')}"
    if max(m_lo, r_lo) > 1.0 and (coin.get("lift") or 0.0) >= max(m_lo, r_lo):
        # A static per-coin number does as well: the signal says which coin,
        # not when.
        return "coin_choice_only"
    if m_lo > 1.0 and m_lo > r_lift:
        return "precursor_found"
    if m_lo > 1.0:
        return f"signal_matched_by:{rule.get('rule')}"
    if r_lo > 1.0 and (rule.get("episodes") or 0) >= MIN_EPISODES:
        return f"single_feature_signal:{rule.get('rule')}"
    return "no_precursor"


def _permutation_factors(head: dict[str, Any], x: Any, y: list[int], top: int = 8) -> list[dict[str, Any]]:
    """Drop in PR-AUC on the test segment when each feature is shuffled."""
    import numpy as np
    from sklearn.metrics import average_precision_score

    base = float(average_precision_score(y, _scores(head, x)))
    if base <= 0:
        return []
    rng = np.random.default_rng(7)
    out = []
    for j, name in enumerate(MODEL_FEATURES):
        drops = []
        for _ in range(2):
            shuffled = x.copy()
            shuffled[:, j] = rng.permutation(shuffled[:, j])
            drops.append(base - float(average_precision_score(y, _scores(head, shuffled))))
        out.append({"feature": name, "importance_pct": 100.0 * (sum(drops) / len(drops)) / base})
    out.sort(key=lambda d: d["importance_pct"], reverse=True)
    return out[:top]


def evaluate_arm(
    arm: _Arm,
    *,
    budget: float = DEFAULT_BUDGET,
    train_frac: float = 0.6,
    calib_frac: float = 0.15,
    z: float = 1.96,
    importance: bool = True,
) -> dict[str, Any]:
    import numpy as np

    from amber.models.train import _fit_head

    if not len(arm.y):
        return {"status": "no_calm_rows"}
    x = np.frombuffer(arm.x, dtype=np.float32).reshape(-1, _N_FEATURES).astype(np.float64)
    y_all = np.frombuffer(arm.y, dtype=np.int8)
    ts_all = np.frombuffer(arm.ts, dtype=np.int64)
    sym_all = np.frombuffer(arm.sym, dtype=np.int16)
    order = np.argsort(ts_all, kind="stable")
    x, y_all, ts_all, sym_all = x[order], y_all[order], ts_all[order], sym_all[order]
    # The label reaches horizon+1 bars past the row; the purge gap must cover it.
    splits = make_holdout_splits(ts_all.tolist(), train_frac=train_frac, calib_frac=calib_frac,
                                 gap=max(30, arm.horizon + 1))
    if splits is None:
        return {"status": "too_small", "rows": int(len(y_all))}
    train = ts_all <= splits["train_end"]
    test = ts_all >= splits["test_start"]
    if not train.any() or not test.any():
        return {"status": "empty_segment", "rows": int(len(y_all))}
    y_train, y_test = y_all[train], y_all[test]
    if y_train.min() == y_train.max() or y_test.min() == y_test.max():
        return {"status": "single_class", "rows": int(len(y_all)),
                "test_positives": int(y_test.sum()), "train_positives": int(y_train.sum())}

    head = _fit_head(x[train], y_train.tolist())
    if head.get("type") == "constant":
        return {"status": "degenerate_model", "rows": int(len(y_all))}

    x_test = x[test]
    y = y_test.tolist()
    ts = ts_all[test].tolist()
    scores = _scores(head, x_test).tolist()
    model_res = _precision_at_budget(scores, y, ts, budget, z, arm.horizon)
    rule_res = _best_rule(x_test, y, ts, budget=budget, z=z, horizon=arm.horizon)
    coin_res, within_res, model_within = _coin_controls(
        x[train], sym_all[train], x_test, sym_all[test], y, ts, scores,
        budget=budget, z=z, horizon=arm.horizon,
    )
    positive_episodes = _episodes([t for t, v in zip(ts, y) if v], arm.horizon)
    out: dict[str, Any] = {
        "status": "ok",
        "window": arm.window,
        "horizon": arm.horizon,
        "rows": int(len(y_all)),
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "test_hours": (ts[-1] - ts[0]) / 3_600_000 if ts else 0.0,
        "test_positives": int(y_test.sum()),
        "positive_episodes": positive_episodes,
        "base_rate": float(y_test.mean()),
        "model": model_res,
        "best_rule": rule_res,
        "coin_only": coin_res,
        "within_coin": within_res,
        "model_within_coin": model_within,
        "verdict": _verdict(model_res, rule_res, positive_episodes, coin_res, within_res, model_within),
    }
    if importance:
        out["top_factors"] = _permutation_factors(head, x_test, y)
    return out


def run_ignition_check(
    features_root: Path,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    calm_pct: float = DEFAULT_CALM_PCT,
    barrier: float = DEFAULT_BARRIER,
    budget: float = DEFAULT_BUDGET,
    max_candles_per_symbol: int = DEFAULT_DAYS * 1440,
    min_warmup_bars: int = 60,
    train_frac: float = 0.6,
    calib_frac: float = 0.15,
) -> dict[str, Any]:
    arms, symbols = collect_arms(
        features_root, windows=windows, horizons=horizons, calm_pct=calm_pct, barrier=barrier,
        max_candles_per_symbol=max_candles_per_symbol, min_warmup_bars=min_warmup_bars,
    )
    if not symbols:
        return {"status": "no_features", "arms": []}

    # Each arm compares the model, every feature in both directions, the
    # per-coin control, and every within-coin feature in both directions.
    z = family_z(max(1, len(arms)) * (3 + 4 * _N_FEATURES))
    results: list[dict[str, Any]] = []
    span_days = 0.0
    for arm in arms:
        if len(arm.ts):
            span_days = max(span_days, (max(arm.ts) - min(arm.ts)) / 86_400_000)
        res = evaluate_arm(arm, budget=budget, train_frac=train_frac, calib_frac=calib_frac, z=z)
        res.update({"window": arm.window, "horizon": arm.horizon,
                    "calm_share": (len(arm.y) / arm.considered) if arm.considered else None})
        logger.info("ignition arm window=%s horizon=%s -> %s", arm.window, arm.horizon,
                    res.get("verdict", res.get("status")))
        results.append(res)

    scored = [a for a in results if a.get("status") == "ok"]
    best = max(scored, key=lambda a: _VERDICT_RANK.get(a["verdict"].split(":")[0], 0), default=None)
    return {
        "status": "ok",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "span_days": span_days,
        "calm_pct": calm_pct,
        "barrier": barrier,
        "budget": budget,
        "z": z,
        "min_positive_episodes": MIN_POSITIVE_EPISODES,
        "arms": results,
        "verdict": best["verdict"] if best else "no_scored_arm",
    }


_VERDICT_RU = {
    "timing_signal": "ПРЕДВЕСТНИК ПО ВРЕМЕНИ: монета ведёт себя необычно для себя самой перед ходом",
    "coin_choice_only": "сигнал только о выборе монеты: «нервные» монеты ходят чаще, но КОГДА — не видно",
    "precursor_found": "НАЙДЕН ПРЕДВЕСТНИК: модель предсказывает ход из тишины лучше любого одного признака",
    "signal_matched_by": "предвестник есть, но один признак ловит его так же хорошо, как модель",
    "single_feature_signal": "слабый предвестник есть только в одном признаке",
    "no_precursor": "предвестника нет: при достаточном числе событий спокойный рынок не предупреждает о ходе",
    "underpowered": "мало данных: событий или эпизодов слишком мало для вывода в любую сторону",
}


def format_report(report: dict[str, Any]) -> str:
    if report.get("status") != "ok":
        return f"ignition check: {report.get('status')}"

    def f(v: Any, spec: str = "{:.3f}") -> str:
        return "-" if v is None else spec.format(v)

    lines = [
        f"Зарождение движения: тишина (цена и свечи high-low < {report['calm_pct'] * 100:.1f}% за окно) "
        f"→ ход ≥{report['barrier'] * 100:.1f}% в любую сторону, вход со следующей свечи",
        f"{report['symbols']} символов · история {report.get('span_days', 0):.1f} дн · бюджет "
        f"{report['budget'] * 100:.1f}% тихих строк теста · z={report['z']:.2f} (поправка на все сравнения)",
        f"вывод только при ≥{report.get('min_positive_episodes', MIN_POSITIVE_EPISODES)} независимых "
        "событиях-ходах в тесте (колонка «событ»)",
        "",
        f"{'окно':>4} {'гориз':>5} {'тихих':>6} {'тест,ч':>6} {'событ':>5} {'база':>6} │ {'модель':>6} {'lift':>5} "
        f"{'lift_lo':>7} {'эпиз':>4} │ {'лучший признак':<22} {'lift':>5} {'lift_lo':>7} │ вывод",
    ]
    for a in report["arms"]:
        if a.get("status") != "ok":
            lines.append(f"{a['window']:>4} {a['horizon']:>5}  {a.get('status')}  "
                         f"(событий в тесте: {a.get('test_positives', '-')})")
            continue
        m, r = a["model"], a["best_rule"]
        lines.append(
            f"{a['window']:>4} {a['horizon']:>5} {f(a['calm_share'], '{:.0%}'):>6} {f(a.get('test_hours'), '{:.0f}'):>6} "
            f"{a.get('positive_episodes', 0):>5} {f(a['base_rate'], '{:.4f}'):>6} │ "
            f"{f(m.get('precision')):>6} {f(m.get('lift'), '{:.2f}'):>5} {f(m.get('lift_ci_low_clustered'), '{:.2f}'):>7} "
            f"{m.get('episodes', 0):>4} │ {str(r.get('rule')):<22} {f(r.get('lift'), '{:.2f}'):>5} "
            f"{f(r.get('lift_ci_low_clustered'), '{:.2f}'):>7} │ {a['verdict']}"
        )
        c, w, mw = a.get("coin_only", {}), a.get("within_coin", {}), a.get("model_within_coin", {})
        lines.append(
            f"{'':>10}только монета: lift {f(c.get('lift'), '{:.2f}')} lo {f(c.get('lift_ci_low_clustered'), '{:.2f}')}"
            f" · модель внутри монеты: lift {f(mw.get('lift'), '{:.2f}')} lo "
            f"{f(mw.get('lift_ci_low_clustered'), '{:.2f}')} · признак внутри монеты: {w.get('rule')} "
            f"lift {f(w.get('lift'), '{:.2f}')} lo {f(w.get('lift_ci_low_clustered'), '{:.2f}')}"
        )
    key = report["verdict"].split(":")[0]
    lines += ["", f"ИТОГ: {report['verdict']} — {_VERDICT_RU.get(key, '')}"]
    best = max((a for a in report["arms"] if a.get("top_factors")),
               key=lambda a: a["model"].get("lift_ci_low_clustered") or 0.0, default=None)
    if best and key in ("precursor_found", "signal_matched_by", "timing_signal"):
        lines += ["", f"Факторы модели (окно {best['window']}, горизонт {best['horizon']}), падение PR-AUC при перемешивании:"]
        lines += [f"  {t['feature']:<20} {t['importance_pct']:7.2f}%" for t in best["top_factors"]]
    elif best:
        lines += ["", "Факторы не показаны: без доказанного сигнала их ранжирование — шум."]
    return "\n".join(lines)


def save_report(logs_root: Path, report: dict[str, Any]) -> Path:
    logs_root.mkdir(parents=True, exist_ok=True)
    path = logs_root / REPORT_FILE
    path.write_text(json.dumps(report, ensure_ascii=False, default=str), encoding="utf-8")
    return path
