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

The feature set is the live model's plus two families added 2026-10-06,
after the 1m features showed only coin choice: BTC/ETH lead (has the market
leader just moved?) and hour-scale build-up (compression, volume). Arms are
pre-registered in DEFAULT_ARMS, including a liquid/thin split by spread and a
6-hour-calm -> 4-hour-move question.

Because the events are rare, this check reads far more history than the
training window (`--days`, 30 by default), held as compact float32 arrays —
dict rows for 30 days x 27 symbols would not fit the box.

Read-only: trains in memory, writes only logs/ignition_check.json.
"""

from __future__ import annotations

from array import array
from collections import deque
from dataclasses import dataclass
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

# Added 2026-10-06, when the 1m model features gave only coin choice
# (`coin_choice_only`). Both families are built from data already on disk.
#
# Market lead: alts often move seconds to minutes AFTER BTC/ETH. "BTC just
# moved, this coin is still calm" is a genuine timing signal if it exists.
LEAD_FEATURES = ("btc_absret_1", "btc_absret_5", "btc_range_atr_14", "eth_absret_5")
# Hour scale: compression and volume build-up over hours, the scale where
# positions are said to be accumulated before a move.
HOUR_FEATURES = ("hr_range_240", "hr_compress_240_1440", "hr_absret_240", "hr_notional_60_1440")
FEATURES = tuple(MODEL_FEATURES) + LEAD_FEATURES + HOUR_FEATURES
_N_FEATURES = len(FEATURES)
_HOUR_CONTEXT = 1440


@dataclass(frozen=True)
class ArmSpec:
    """One pre-registered question: calm for `window` bars (range under
    `calm_pct`) -> a `barrier` move within `horizon` bars, evaluated on each
    coin group in `groups` ("all", "liquid", "thin")."""

    name: str
    window: int
    horizon: int
    calm_pct: float
    barrier: float
    groups: tuple[str, ...] = ("all",)


# Fixed 2026-10-06 before the run they were added for. The 15-bar horizons of
# the first runs are dropped: on 30 days they held 17-44 moves, under or near
# the floor, and every extra arm raises the bar for all of them.
DEFAULT_ARMS = (
    ArmSpec("m30", 30, 30, DEFAULT_CALM_PCT, DEFAULT_BARRIER, ("all", "liquid", "thin")),
    ArmSpec("m60", 60, 30, DEFAULT_CALM_PCT, DEFAULT_BARRIER),
    # Calm for 6 hours (range < 1.5%) -> a 3% move within 4 hours.
    ArmSpec("h6", 360, 240, 0.015, 0.03),
)

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


def _rolling(values: Sequence[float], window: int, take_max: bool) -> list[float]:
    """out[i] = max (or min) of values[max(0, i-window+1) : i+1], in O(n)."""
    out: list[float] = []
    q: deque[int] = deque()
    for i, v in enumerate(values):
        while q and ((values[q[-1]] <= v) if take_max else (values[q[-1]] >= v)):
            q.pop()
        q.append(i)
        if q[0] <= i - window:
            q.popleft()
        out.append(values[q[0]])
    return out


class _Rolling:
    """Rolling max/min of one price series, computed once per window."""

    def __init__(self, prices: Sequence[float]) -> None:
        self.prices = prices
        self._cache: dict[tuple[int, bool], list[float]] = {}

    def get(self, window: int, take_max: bool) -> list[float]:
        key = (window, take_max)
        if key not in self._cache:
            self._cache[key] = _rolling(self.prices, window, take_max)
        return self._cache[key]


def calm_now(
    roll: "_Rolling",
    synthetic_prefix: Sequence[int],
    candle_ranges: Sequence[float],
    i: int,
    *,
    window: int,
    calm_pct: float,
) -> bool:
    """Is bar `i` calm, judged only on bars up to and including `i`?

    The single definition shared by the check (which adds the forward window
    on top) and the live scanner (which cannot see forward). A second copy
    here is how a live channel would quietly score a different population
    from the one that was validated.
    """
    lo = i - window + 1
    if lo < 0 or synthetic_prefix[i + 1] - synthetic_prefix[lo] > 0:
        return False
    prices = roll.prices
    p_now = prices[i]
    pmin = roll.get(window, False)[i]
    if p_now <= 0 or pmin <= 0:
        return False
    if (roll.get(window, True)[i] - pmin) / p_now > calm_pct:
        return False
    if (roll.get(RECENT_BARS, True)[i] - roll.get(RECENT_BARS, False)[i]) / p_now > calm_pct * RECENT_FRACTION:
        return False
    ends = set(range(i, lo + CANDLE_SLICE - 2, -CANDLE_SLICE)) | {lo + CANDLE_SLICE - 1}
    return not any(candle_ranges[e] >= calm_pct for e in ends if 0 <= e <= i)


def _calm_labels_fast(
    roll: _Rolling,
    synthetic_prefix: Sequence[int],
    candidates: Sequence[int],
    *,
    window: int,
    horizon: int,
    calm_pct: float,
    barrier: float,
    candle_ranges: Sequence[float],
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Same result as `_calm_labels`, with every window looked up in O(1).

    The row-by-row version slices the window for each bar, which is fine at
    30-60 bars and takes about an hour at the 6-hour arm. A test pins the two
    to identical output on random series with gaps and wicks.
    """
    assert window >= RECENT_BARS
    prices = roll.prices
    n = len(prices)
    fmax, fmin = roll.get(horizon, True), roll.get(horizon, False)
    out: list[tuple[int, int, int, int]] = []
    considered = 0
    for i in candidates:
        lo, hi = i - window + 1, i + 1 + horizon
        if lo < 0 or hi >= n:
            continue
        considered += 1
        if synthetic_prefix[hi + 1] - synthetic_prefix[lo] > 0:
            continue  # a gap in the forward window would read as "no move"
        if not calm_now(roll, synthetic_prefix, candle_ranges, i, window=window, calm_pct=calm_pct):
            continue
        entry = prices[i + 1]
        # Bars i+2 .. i+1+horizon, measured from the entry bar, as label_path.
        up = int(fmax[hi] >= entry * (1.0 + barrier))
        down = int(fmin[hi] <= entry * (1.0 - barrier))
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
    """Calm rows for one ArmSpec, as compact arrays."""

    def __init__(self, spec: ArmSpec) -> None:
        self.spec = spec
        self.window, self.horizon = spec.window, spec.horizon
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


def _lead_table(features_dir: Path, symbol: str, want: int) -> dict[int, tuple[float, float, float]]:
    """ts -> (|ret 1 bar|, |ret 5 bars|, range_atr_14) for a market leader."""
    d = features_dir / symbol
    if not d.is_dir():
        return {}
    return lead_from_rows(_load_symbol(d, want))


def lead_from_rows(rows: list[dict[str, Any]]) -> dict[int, tuple[float, float, float]]:
    prices = [float(r.get("mid_price", 0.0) or 0.0) for r in rows]
    out: dict[int, tuple[float, float, float]] = {}
    for i, r in enumerate(rows):
        p = prices[i]
        a1 = abs(p / prices[i - 1] - 1.0) if i >= 1 and prices[i - 1] > 0 else 0.0
        a5 = abs(p / prices[i - 5] - 1.0) if i >= 5 and prices[i - 5] > 0 else 0.0
        out[int(r.get("ts", 0) or 0)] = (a1, a5, float(r.get("range_atr_14", 0.0) or 0.0))
    return out


def _hour_features(prices: Sequence[float], notional: Sequence[float], roll: _Rolling) -> list[tuple[float, ...]]:
    """Per bar: range over 4 h, its share of the 24 h range (compression), the
    4 h return, and the last hour's notional against the 24 h hourly mean."""
    mx240, mn240 = roll.get(240, True), roll.get(240, False)
    mx1440, mn1440 = roll.get(_HOUR_CONTEXT, True), roll.get(_HOUR_CONTEXT, False)
    prefix = [0.0]
    for v in notional:
        prefix.append(prefix[-1] + v)
    out = []
    for i, p in enumerate(prices):
        if p <= 0:
            out.append((0.0, 0.0, 0.0, 0.0))
            continue
        r240 = (mx240[i] - mn240[i]) / p
        r1440 = (mx1440[i] - mn1440[i]) / p
        ret240 = abs(p / prices[i - 240] - 1.0) if i >= 240 and prices[i - 240] > 0 else 0.0
        last_hour = prefix[i + 1] - prefix[max(0, i - 59)]
        day = prefix[i + 1] - prefix[max(0, i - _HOUR_CONTEXT + 1)]
        hours = min(i + 1, _HOUR_CONTEXT) / 60.0
        ratio = last_hour / (day / hours) if day > 0 else 0.0
        out.append((r240, (r240 / r1440) if r1440 > 0 else 0.0, ret240, ratio))
    return out


def feature_vector(
    row: dict[str, Any],
    btc: dict[int, tuple[float, float, float]],
    eth: dict[int, tuple[float, float, float]],
    hour: tuple[float, ...],
) -> list[float]:
    """The row in FEATURES order. Training and live scoring both build it here."""
    ts = int(row.get("ts", 0) or 0)
    b = btc.get(ts, (0.0, 0.0, 0.0))
    e = eth.get(ts, (0.0, 0.0, 0.0))
    return ([float(row.get(name, 0.0) or 0.0) for name in MODEL_FEATURES]
            + [b[0], b[1], b[2], e[1]] + list(hour))


def collect_arms(
    features_root: Path,
    *,
    specs: Sequence[ArmSpec],
    max_candles_per_symbol: int,
    min_warmup_bars: int,
) -> tuple[list[_Arm], int, list[str], list[float]]:
    """One pass over the feature files, one symbol in memory at a time.

    Returns the arms, the number of symbols read, their names, and each
    symbol's mean spread (for the liquid/thin split).
    """
    arms = [_Arm(spec) for spec in specs]
    features_dir = features_root / "features"
    if not features_dir.exists():
        return arms, 0, [], []
    context = max(max(sp.window for sp in specs), _HOUR_CONTEXT) + 1
    if max_candles_per_symbol <= 0:
        max_candles_per_symbol = 10**9  # all history
    want = max_candles_per_symbol + context
    btc = _lead_table(features_dir, "BTCUSDT", want)
    eth = _lead_table(features_dir, "ETHUSDT", want)
    names: list[str] = []
    spreads: list[float] = []
    for symbol_dir in sorted(p for p in features_dir.iterdir() if p.is_dir()):
        rows = _load_symbol(symbol_dir, want)
        if len(rows) < min_warmup_bars + 60:
            continue
        sym_idx = len(names)
        names.append(symbol_dir.name)
        spreads.append(sum(float(r.get("spread_bps", 0.0) or 0.0) for r in rows) / len(rows))
        prices = [float(r.get("mid_price", 0.0) or 0.0) for r in rows]
        synthetic_prefix = [0]
        for r in rows:
            synthetic_prefix.append(synthetic_prefix[-1] + int(bool(r.get("is_synthetic", False))))
        ranges = [candle_range_20(r) for r in rows]
        roll = _Rolling(prices)
        hours = _hour_features(prices, [float(r.get("notional_volume_1m", 0.0) or 0.0) for r in rows], roll)
        start = max(0, len(rows) - max_candles_per_symbol)
        candidates = [
            i for i in range(start, len(rows))
            if not rows[i].get("is_synthetic", False) and int(rows[i].get("obs", 0) or 0) >= min_warmup_bars
        ]
        for arm in arms:
            sp = arm.spec
            labelled, considered = _calm_labels_fast(
                roll, synthetic_prefix, candidates, window=sp.window, horizon=sp.horizon,
                calm_pct=sp.calm_pct, barrier=sp.barrier, candle_ranges=ranges,
            )
            arm.considered += considered
            for i, move, _up, _down in labelled:
                arm.x.extend(feature_vector(rows[i], btc, eth, hours[i]))
                arm.y.append(move)
                arm.ts.append(int(rows[i].get("ts", 0) or 0))
                arm.sym.append(sym_idx)
        del rows, prices, synthetic_prefix, ranges, roll, hours
    return arms, len(names), names, spreads


def _scores(head: dict[str, Any], x: Any) -> Any:
    import numpy as np

    if head.get("type") == "lightgbm":
        import lightgbm as lgb

        return lgb.Booster(model_str=head["booster"]).predict(x)
    if head.get("type") == "logreg":
        w = np.asarray([head["weights"].get(n, 0.0) for n in FEATURES], dtype=float)
        return 1.0 / (1.0 + np.exp(-(x @ w + float(head.get("bias", 0.0)))))
    return np.full(len(x), float(head.get("prob", 0.0)))


def _best_rule(x: Any, y: list[int], ts: list[int], *, budget: float, z: float, horizon: int) -> dict[str, Any]:
    """Every feature as a one-line rule, both directions, best achieved lift.

    Chosen on the test segment itself, which flatters the rule — deliberately:
    the model has to beat the luckiest single feature, not a fair one.
    """
    best: dict[str, Any] | None = None
    for j, name in enumerate(FEATURES):
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

    j_atr = list(FEATURES).index("range_atr_14")
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
    for j, name in enumerate(FEATURES):
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
    symbols: Sequence[int] | None = None,
) -> dict[str, Any]:
    import numpy as np

    from amber.models.train import _fit_head

    if not len(arm.y):
        return {"status": "no_calm_rows"}
    x = np.frombuffer(arm.x, dtype=np.float32).reshape(-1, _N_FEATURES).astype(np.float64)
    y_all = np.frombuffer(arm.y, dtype=np.int8)
    ts_all = np.frombuffer(arm.ts, dtype=np.int64)
    sym_all = np.frombuffer(arm.sym, dtype=np.int16)
    if symbols is not None:
        keep = np.isin(sym_all, np.asarray(list(symbols), dtype=np.int16))
        x, y_all, ts_all, sym_all = x[keep], y_all[keep], ts_all[keep], sym_all[keep]
        if not len(y_all):
            return {"status": "no_calm_rows"}
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
    arms: Sequence[ArmSpec] = DEFAULT_ARMS,
    budget: float = DEFAULT_BUDGET,
    max_candles_per_symbol: int = DEFAULT_DAYS * 1440,
    min_warmup_bars: int = 60,
    train_frac: float = 0.6,
    calib_frac: float = 0.15,
) -> dict[str, Any]:
    collected, n_symbols, names, spreads = collect_arms(
        features_root, specs=arms, max_candles_per_symbol=max_candles_per_symbol,
        min_warmup_bars=min_warmup_bars,
    )
    if not n_symbols:
        return {"status": "no_features", "arms": []}

    # Liquid = mean spread at or below the median coin's; thin = above it.
    median = sorted(spreads)[len(spreads) // 2] if spreads else 0.0
    groups = {
        "all": None,
        "liquid": [i for i, sp in enumerate(spreads) if sp <= median],
        "thin": [i for i, sp in enumerate(spreads) if sp > median],
    }
    n_evals = sum(len(a.spec.groups) for a in collected)
    # Each evaluation compares the model, every feature in both directions,
    # the per-coin control, the model within coin, and every within-coin
    # feature in both directions.
    z = family_z(max(1, n_evals) * (3 + 4 * _N_FEATURES))
    results: list[dict[str, Any]] = []
    span_days = 0.0
    for arm in collected:
        sp = arm.spec
        if len(arm.ts):
            span_days = max(span_days, (max(arm.ts) - min(arm.ts)) / 86_400_000)
        for group in sp.groups:
            res = evaluate_arm(arm, budget=budget, train_frac=train_frac, calib_frac=calib_frac, z=z,
                               symbols=groups[group])
            res.update({
                "name": sp.name, "group": group, "window": sp.window, "horizon": sp.horizon,
                "calm_pct": sp.calm_pct, "barrier": sp.barrier,
                "calm_share": (len(arm.y) / arm.considered) if arm.considered else None,
            })
            logger.info("ignition arm %s/%s -> %s", sp.name, group, res.get("verdict", res.get("status")))
            results.append(res)

    scored = [a for a in results if a.get("status") == "ok"]
    best = max(scored, key=lambda a: _VERDICT_RANK.get(a["verdict"].split(":")[0], 0), default=None)
    return {
        "status": "ok",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "symbols": n_symbols,
        "groups": {g: [names[i] for i in idx] for g, idx in groups.items() if idx is not None},
        "span_days": span_days,
        "budget": budget,
        "z": z,
        "min_positive_episodes": MIN_POSITIVE_EPISODES,
        "arms": results,
        "verdict": best["verdict"] if best else "no_scored_arm",
        "verdict_arm": f"{best['name']}/{best['group']}" if best else None,
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
        "Зарождение движения: рынок спокоен (цена и свечи) → резкий ход в любую сторону, вход со следующей свечи",
        f"{report['symbols']} символов · история {report.get('span_days', 0):.1f} дн · бюджет "
        f"{report['budget'] * 100:.1f}% тихих строк теста · z={report['z']:.2f} (поправка на все сравнения)",
        f"вывод только при ≥{report.get('min_positive_episodes', MIN_POSITIVE_EPISODES)} независимых "
        "событиях-ходах в тесте (колонка «событ»)",
        "m30/m60: тишина 30/60 мин (<0.5%) → ход ≥1% за 30 мин · h6: тишина 6 ч (<1.5%) → ход ≥3% за 4 ч",
        "",
        f"{'вариант':<11} {'тихих':>6} {'тест,ч':>6} {'событ':>5} {'база':>6} │ {'модель':>6} {'lift':>5} "
        f"{'lift_lo':>7} {'эпиз':>4} │ {'лучший признак':<24} {'lift':>5} {'lift_lo':>7} │ вывод",
    ]
    for a in report["arms"]:
        label = f"{a.get('name')}/{a.get('group')}"
        if a.get("status") != "ok":
            lines.append(f"{label:<11}  {a.get('status')}  (событий в тесте: {a.get('test_positives', '-')})")
            continue
        m, r = a["model"], a["best_rule"]
        lines.append(
            f"{label:<11} {f(a['calm_share'], '{:.0%}'):>6} {f(a.get('test_hours'), '{:.0f}'):>6} "
            f"{a.get('positive_episodes', 0):>5} {f(a['base_rate'], '{:.4f}'):>6} │ "
            f"{f(m.get('precision')):>6} {f(m.get('lift'), '{:.2f}'):>5} {f(m.get('lift_ci_low_clustered'), '{:.2f}'):>7} "
            f"{m.get('episodes', 0):>4} │ {str(r.get('rule')):<24} {f(r.get('lift'), '{:.2f}'):>5} "
            f"{f(r.get('lift_ci_low_clustered'), '{:.2f}'):>7} │ {a['verdict']}"
        )
        c, w, mw = a.get("coin_only", {}), a.get("within_coin", {}), a.get("model_within_coin", {})
        lines.append(
            f"{'':>12}только монета: lift {f(c.get('lift'), '{:.2f}')} lo {f(c.get('lift_ci_low_clustered'), '{:.2f}')}"
            f" · модель внутри монеты: lift {f(mw.get('lift'), '{:.2f}')} lo "
            f"{f(mw.get('lift_ci_low_clustered'), '{:.2f}')} · признак внутри монеты: {w.get('rule')} "
            f"lift {f(w.get('lift'), '{:.2f}')} lo {f(w.get('lift_ci_low_clustered'), '{:.2f}')}"
        )
    groups = report.get("groups", {})
    if groups.get("thin"):
        lines += ["", "тонкие монеты (спред выше медианы): " + ", ".join(groups["thin"])]
    key = report["verdict"].split(":")[0]
    lines += ["", f"ИТОГ: {report['verdict']} ({report.get('verdict_arm')}) — {_VERDICT_RU.get(key, '')}"]
    best = max((a for a in report["arms"] if a.get("top_factors")),
               key=lambda a: a["model"].get("lift_ci_low_clustered") or 0.0, default=None)
    if best and key in ("precursor_found", "signal_matched_by", "timing_signal"):
        lines += ["", f"Факторы модели ({best['name']}/{best['group']}), падение PR-AUC при перемешивании:"]
        lines += [f"  {t['feature']:<22} {t['importance_pct']:7.2f}%" for t in best["top_factors"]]
    elif best:
        lines += ["", "Факторы не показаны: без доказанного сигнала их ранжирование — шум."]
    return "\n".join(lines)


def save_report(logs_root: Path, report: dict[str, Any]) -> Path:
    logs_root.mkdir(parents=True, exist_ok=True)
    path = logs_root / REPORT_FILE
    path.write_text(json.dumps(report, ensure_ascii=False, default=str), encoding="utf-8")
    return path
