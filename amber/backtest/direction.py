"""Does the order book say which way a move will go? (roadmap D2, plan K)

Every test so far found that Amber can say *that* a 1% move is coming more
often than chance, and nothing could say *which way*; every symmetric trading
rule then paid the fee on a coin flip (CLAUDE.md section 10). The book
collector (amber/pipeline/book_app.py) has recorded what 1m bars cannot show:
the resting book near the price and the taker flow inside the minute. This
asks whether that data, at the close of a minute, points to the side of the
next 1% move.

Everything below was fixed on 2026-11-15, before the check saw real data.

Events. Every minute t of a symbol with a full candle window: enter on the
close of minute t+1 (one bar for the human, as everywhere) and watch the
candle wicks for 30 minutes. The label is the side of +-1% touched first, or
0 when neither is touched; a minute whose first touch hits both sides in one
bar is dropped (order unknown), as are windows with a missing or gap-filled
candle. Direction accuracy is measured on the minutes that moved; the share
that moved is kept, because it decides whether a call can pay (Money).

Two populations (`arm`):
- `all`: every such minute: the general direction question.
- `calm`: calm minutes the ignition ledger scored (the warning population),
  their outcome taken from the ledger itself.

Features at the close of minute t, from book rows with time <= that close
(a row is stamped with the end of its 5 s interval), each oriented so that a
positive value says "up":
- `obi_l1`: (bid1 - ask1) / (bid1 + ask1) notional at the last row.
- `obi_5`, `obi_10`: the same within 5 / 10 bps of the mid (the 200-level
  book reaches only ~4-9 bps on BTC/ETH, so there it is the visible book).
- `obi_5_1m`: `obi_5` averaged over the minute's rows.
- `flow_1m`, `flow_5m`: (buy - sell) / (buy + sell) taker notional over the
  last 1 / 5 minutes.
- `obi_10_d5`: change of `obi_10` over the last 5 minutes.
A minute needs >= 10 of its 12 rows with a book, or it is not scored.

Comparisons. Each feature is read in both directions (a positive value says
up, or says down), on all events and on the 20% with the largest |value|;
plus a logistic model on all seven, fitted on the first 70% of the period and
scored on the rest. 2 arms x 2 cuts x (7 x 2 + 1) = 60 comparisons, so every
bound uses family_z(60). Direction accuracy is compared with always calling
the more frequent side on the same events (rule 2: the winner's lower bound
must clear the loser's achieved value). Events within 30 minutes of each
other, on any symbol, are one episode.

Money (an estimate, not a verdict). A call is a trade with stop and target
both 1% and 15 bps of costs. Of the minutes a rule calls, only a share q
reaches either barrier; the rest time out near zero and still pay the cost.
So a call pays only if q x (2p - 1) x 1% > 0.15%, i.e. accuracy p above
0.5 + 0.075 / q: 57.5% when every call moves, 65% at q = 0.5, never below
q = 0.15 (the ~6% of warnings that move included). A signal whose lower bound clears that is
marked `tradeable`; a forward ledger rule would still have to confirm it.
(Corrected 2026-11-15 before the first real run: the first version assumed
q = 1 and would have called 57.5% tradeable for a rule whose calls mostly
never move.)
"""

from __future__ import annotations

from array import array
import gzip
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MINUTE = 60_000
ROW_MS = 5_000
HORIZON = 30
LAG = 1
BARRIER = 0.01
MIN_ROWS = 10
FEATURES = ("obi_l1", "obi_5", "obi_10", "obi_5_1m", "flow_1m", "flow_5m", "obi_10_d5")
ARMS = ("all", "calm")
CUTS = ("all", "top20")
TOP_SHARE = 0.20
FIT_SHARE = 0.70
MIN_EPISODES = 30
COST = 0.0015
BREAK_EVEN = 0.5 + COST / (2 * BARRIER)  # 57.5% if every call reached a barrier
N_COMPARISONS = len(ARMS) * len(CUTS) * (2 * len(FEATURES) + 1)

_COLS = ("ts", "mid", "bid1_usd", "ask1_usd", "bid5_usd", "ask5_usd", "bid10_usd", "ask10_usd",
         "buy_usd", "sell_usd")


def _imb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    s = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, (a - b) / s, np.nan)


def read_book(symbol_dir: Path, since_ms: int = 0) -> dict[str, np.ndarray]:
    """One symbol's stored rows as arrays, oldest first (csv and csv.gz)."""
    from amber.exchange.orderbook import COLUMNS

    idx = [COLUMNS.index(c) for c in _COLS]
    # array('d'): 8 bytes a value instead of a float object's ~32 — a symbol's
    # month of 5 s rows is ~500k x 10 values.
    out = [array("d") for _ in _COLS]
    for path in sorted(Path(symbol_dir).glob("*.csv*")):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                cells = line.rstrip("\n").split(",")
                if len(cells) != len(COLUMNS) or not cells[0].isdigit():
                    continue  # header or a torn last line
                ts = int(cells[0])
                if ts < since_ms:
                    continue
                for k, i in enumerate(idx):
                    c = cells[i]
                    out[k].append(float(c) if c else math.nan)
    arr = {c: np.frombuffer(v, dtype=np.float64).copy() if len(v) else np.zeros(0) for c, v in zip(_COLS, out)}
    order = np.argsort(arr["ts"], kind="stable")
    return {c: v[order] for c, v in arr.items()}


def minute_features(book: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    """Features per minute open time, from rows stamped (open, open + 1 min]."""
    ts = book["ts"]
    if ts.size == 0:
        return {}
    minute = ((ts.astype(np.int64) - 1) // MINUTE) * MINUTE
    has_book = ~np.isnan(book["mid"])
    obi_l1 = _imb(book["bid1_usd"], book["ask1_usd"])
    obi_5 = _imb(book["bid5_usd"], book["ask5_usd"])
    obi_10 = _imb(book["bid10_usd"], book["ask10_usd"])
    buy = np.nan_to_num(book["buy_usd"])
    sell = np.nan_to_num(book["sell_usd"])

    keys, start = np.unique(minute, return_index=True)
    end = np.append(start[1:], ts.size)
    n = keys.size
    rows_ok = np.add.reduceat(has_book.astype(np.int64), start)
    buy_m = np.add.reduceat(buy, start)
    sell_m = np.add.reduceat(sell, start)
    last = end - 1
    # An empty 5 bps band on both sides (a spread wider than 10 bps) has no
    # imbalance; average over the rows that have one, so one such row does
    # not blank the whole minute.
    o5_ok = has_book & np.isfinite(obi_5)
    o5_n = np.add.reduceat(o5_ok.astype(np.int64), start)
    with np.errstate(invalid="ignore", divide="ignore"):
        o5_mean = np.where(o5_n > 0, np.add.reduceat(np.where(o5_ok, obi_5, 0.0), start) / o5_n, np.nan)

    # 5-minute flow and obi_10 change need the minutes before; minutes are
    # contiguous only where data is, so look them up by key.
    pos = {int(k): i for i, k in enumerate(keys)}
    feats: dict[int, np.ndarray] = {}
    for i in range(n):
        if rows_ok[i] < MIN_ROWS or not has_book[last[i]]:
            continue
        k = int(keys[i])
        b5 = s5 = 0.0
        for j in range(5):
            p = pos.get(k - j * MINUTE)
            if p is not None:
                b5 += buy_m[p]
                s5 += sell_m[p]
        prev = pos.get(k - 5 * MINUTE)
        d5 = math.nan
        if prev is not None and has_book[last[prev]]:
            d5 = obi_10[last[i]] - obi_10[last[prev]]
        tot1, tot5 = buy_m[i] + sell_m[i], b5 + s5
        feats[k] = np.array([
            obi_l1[last[i]], obi_5[last[i]], obi_10[last[i]], o5_mean[i],
            (buy_m[i] - sell_m[i]) / tot1 if tot1 > 0 else math.nan,
            (b5 - s5) / tot5 if tot5 > 0 else math.nan,
            d5,
        ], dtype=np.float64)
    return feats


def candle_labels(candles: list[dict[str, Any]], start_ms: int = 0) -> dict[int, int]:
    """First touch (+1 / -1) of +-1% after entering on the close of bar t+1; 0 if none."""
    if not candles:
        return {}
    ts = np.array([int(c.get("ts", 0) or 0) for c in candles], dtype=np.int64)
    hi = np.array([float(c.get("high", 0) or 0) for c in candles])
    lo = np.array([float(c.get("low", 0) or 0) for c in candles])
    cl = np.array([float(c.get("close", 0) or 0) for c in candles])
    syn = np.array([bool(c.get("is_synthetic", False)) for c in candles])
    n = ts.size
    span = LAG + HORIZON
    out: dict[int, int] = {}
    if n <= span:
        return out
    m = n - span
    # A window is usable only if every bar in it is real and one minute apart.
    good = (np.diff(ts) == MINUTE) & ~syn[1:]
    ok_run = np.ones(m, dtype=bool)
    for k in range(span):
        ok_run &= good[k:k + m]
    entry = cl[LAG:LAG + m]
    up_lvl, dn_lvl = entry * (1 + BARRIER), entry * (1 - BARRIER)
    first_up = np.full(m, -1)
    first_dn = np.full(m, -1)
    for k in range(LAG + 1, span + 1):
        u = (hi[k:k + m] >= up_lvl) & (first_up < 0)
        d = (lo[k:k + m] <= dn_lvl) & (lo[k:k + m] > 0) & (first_dn < 0)
        first_up[u] = k
        first_dn[d] = k
    for i in np.nonzero(ok_run & (entry > 0))[0]:
        if ts[i] < start_ms:
            continue
        fu, fd = first_up[i], first_dn[i]
        if fu < 0 and fd < 0:
            out[int(ts[i])] = 0  # no move: a call here times out and pays the cost
        elif fu > 0 and (fd < 0 or fu < fd):
            out[int(ts[i])] = 1
        elif fd > 0 and (fu < 0 or fd < fu):
            out[int(ts[i])] = -1
        # equal: both inside one bar, order unknown -> dropped
    return out


def ledger_labels(ledger_path: Path, start_ms: int = 0) -> dict[str, dict[int, int]]:
    """Calm minutes from the ignition ledger: first touch +1 / -1, or 0 for no move."""
    out: dict[str, dict[int, int]] = {}
    if not Path(ledger_path).exists():
        return out
    with Path(ledger_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") != "ok":
                continue
            ft = r.get("first_touch")
            if ft in (1, -1):
                lab = int(ft)
            elif ft is None and not int(r.get("move_hit", 0) or 0):
                lab = 0
            else:
                continue  # both sides in one bar: order unknown
            ts = int(r.get("event_ts", 0) or 0)
            if ts >= start_ms:
                out.setdefault(str(r["symbol"]), {})[ts] = lab
    return out


def break_even(q: float) -> float | None:
    """Accuracy a call needs to pay when a share q of calls reaches a barrier."""
    if q <= 0:
        return None
    p = 0.5 + COST / (2 * q * BARRIER)
    return p if p < 1 else None


def _evaluate(x: np.ndarray, y: np.ndarray, ts: np.ndarray, z: float) -> dict[str, Any]:
    """Accuracy of sign(x) as a call on the minutes that moved (y = +-1),
    against always calling their majority side; q = the share of called
    minutes that moved at all (y = 0 is no move)."""
    from amber.backtest.label_sweep import _episodes, clustered_low

    keep = np.isfinite(x) & (x != 0)
    called = int(keep.sum())
    moved = keep & (y != 0)
    x, y, ts = x[moved], y[moved], ts[moved]
    q = (y.size / called) if called else 0.0
    if y.size == 0:
        return {"called": called, "events": 0, "episodes": 0, "accuracy": None, "base": None, "low": None,
                "move_share": q, "break_even": break_even(q)}
    acc = float(np.mean(np.sign(x) == y))
    base = float(max(np.mean(y == 1), np.mean(y == -1)))
    eps = _episodes([int(t) for t in ts], HORIZON)
    return {"called": called, "events": int(y.size), "episodes": eps, "accuracy": acc, "base": base,
            "low": clustered_low(acc, eps, z) if eps else None, "move_share": q, "break_even": break_even(q)}


def _top(x: np.ndarray, share: float = TOP_SHARE) -> np.ndarray:
    a = np.abs(x)
    fin = np.isfinite(a)
    if not fin.any():
        return np.zeros_like(x, dtype=bool)
    cut = np.quantile(a[fin], 1 - share)
    return fin & (a >= cut)


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.linear_model import LogisticRegression

    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    xs = np.nan_to_num((x - mu) / sd)
    model = LogisticRegression(C=1.0, max_iter=500).fit(xs, (y == 1).astype(int))
    return model.coef_[0], np.array([model.intercept_[0]]), np.vstack([mu, sd])


def evaluate_arm(x: np.ndarray, y: np.ndarray, ts: np.ndarray, *, z: float) -> list[dict[str, Any]]:
    out = []
    for cut in CUTS:
        for j, name in enumerate(FEATURES):
            for sign in (1, -1):
                f = sign * x[:, j]
                mask = _top(f) if cut == "top20" else np.ones(f.size, dtype=bool)
                r = _evaluate(f[mask], y[mask], ts[mask], z)
                out.append({"rule": f"{'+' if sign > 0 else '-'}{name}", "cut": cut, **r})
    # The model: fitted on the first 70% of the period, judged on the rest.
    split = np.quantile(ts, FIT_SHARE) if ts.size else 0
    fit, test = (ts < split) & (y != 0), ts >= split  # it learns the side from minutes that moved
    res = {"called": 0, "events": 0, "episodes": 0, "accuracy": None, "base": None, "low": None,
           "move_share": 0.0, "break_even": None}
    model_rows = []
    if fit.sum() >= 50 and len(set(y[fit].tolist())) == 2 and test.any():
        coef, icpt, norm = _fit_logistic(x[fit], y[fit])
        xs = np.nan_to_num((x[test] - norm[0]) / norm[1])
        score = xs @ coef + icpt[0]
        for cut in CUTS:
            mask = _top(score) if cut == "top20" else np.ones(score.size, dtype=bool)
            model_rows.append({"rule": "model", "cut": cut,
                               **_evaluate(score[mask], y[test][mask], ts[test][mask], z)})
    else:
        model_rows = [{"rule": "model", "cut": c, **res} for c in CUTS]
    return out + model_rows


def _tradeable(row: dict[str, Any]) -> bool:
    """A direction signal whose lower bound clears its own break-even (Money)."""
    be = row.get("break_even")
    return bool(row.get("verdict") == "direction_signal" and be is not None and row["low"] > be)


def _verdict(row: dict[str, Any]) -> str:
    if row["episodes"] < MIN_EPISODES or row["low"] is None:
        return "underpowered"
    if row["low"] > row["base"]:
        return "direction_signal"
    return "no_direction"


def run_direction_check(raw_root: Path, logs_root: Path, symbols: list[str]) -> dict[str, Any]:
    """Read every symbol's book and candles, one at a time; evaluate both arms."""
    from amber.backtest.label_sweep import family_z
    from amber.monitoring.quality_report import _CandleIndex

    book_root = Path(raw_root) / "book"
    calm = ledger_labels(Path(logs_root) / "ignition_ledger.jsonl")
    # One block per symbol and arm: a million events held as tiny per-event
    # arrays cost 744 MB at production size; as blocks they cost ~60 MB.
    xs: dict[str, list[np.ndarray]] = {a: [] for a in ARMS}
    ys: dict[str, list[np.ndarray]] = {a: [] for a in ARMS}
    tss: dict[str, list[np.ndarray]] = {a: [] for a in ARMS}
    span: list[int] = []
    for sym in symbols:
        if not (book_root / sym).exists():
            continue
        feats = minute_features(read_book(book_root / sym))
        if not feats:
            continue
        first, last = min(feats), max(feats)
        span += [first, last]
        need = int((max(last, first) - first) // MINUTE) + LAG + HORIZON + 200
        index = _CandleIndex(Path(raw_root), max_rows=need + 5000)
        labels = {"all": candle_labels(index.candles(sym), start_ms=first), "calm": calm.get(sym, {})}
        for arm in ARMS:
            hits = [(int(t), lab) for t, lab in labels[arm].items() if int(t) in feats]
            if hits:
                xs[arm].append(np.vstack([feats[t] for t, _ in hits]))
                ys[arm].append(np.fromiter((lab for _, lab in hits), dtype=np.int8, count=len(hits)))
                tss[arm].append(np.fromiter((t for t, _ in hits), dtype=np.int64, count=len(hits)))
        del feats, index, labels
    z = family_z(N_COMPARISONS)
    report: dict[str, Any] = {"z": z, "comparisons": N_COMPARISONS, "break_even": BREAK_EVEN,
                              "min_episodes": MIN_EPISODES, "arms": {},
                              "period": (min(span), max(span)) if span else None}
    for arm in ARMS:
        if not ys[arm]:
            report["arms"][arm] = {"minutes": 0, "events": 0, "rows": []}
            continue
        x = np.vstack(xs[arm])
        y = np.concatenate(ys[arm]).astype(np.int64)
        ts = np.concatenate(tss[arm])
        rows = evaluate_arm(x, y, ts, z=z)
        for r in rows:
            r["verdict"] = _verdict(r)
            r["tradeable"] = _tradeable(r)
        moved = y != 0
        report["arms"][arm] = {"minutes": int(y.size), "events": int(moved.sum()),
                               "move_share": float(moved.mean()),
                               "up_share": float(np.mean(y[moved] == 1)) if moved.any() else 0.0, "rows": rows}
    found = [f"{arm}:{r['cut']}:{r['rule']}" for arm in ARMS for r in report["arms"][arm].get("rows", [])
             if r["verdict"] == "direction_signal"]
    powered = any(r["verdict"] != "underpowered" for arm in ARMS for r in report["arms"][arm].get("rows", []))
    report["signals"] = found
    report["tradeable"] = [f"{arm}:{r['cut']}:{r['rule']}" for arm in ARMS
                           for r in report["arms"][arm].get("rows", []) if r.get("tradeable")]
    report["verdict"] = "direction_signal" if found else ("no_direction" if powered else "underpowered")
    return report


def format_report(r: dict[str, Any]) -> str:
    from datetime import datetime, timezone

    def pct(v):
        return "-" if v is None else f"{v * 100:.1f}%"

    lines = ["Направление по стакану и потоку сделок (D2): какой барьер ±1% коснётся первым за 30 мин"]
    if r.get("period"):
        a, b = (datetime.fromtimestamp(v / 1000, tz=timezone.utc).strftime("%m-%d") for v in r["period"])
        lines.append(f"данные стакана: {a} — {b} · сравнений {r['comparisons']} (z={r['z']:.2f})")
    for arm in ARMS:
        a = r["arms"][arm]
        name = {"all": "все минуты перед ходом", "calm": "спокойные минуты (как у предупреждений)"}[arm]
        if not a.get("events"):
            lines.append(f"\n{name}: " + (f"минут {a['minutes']}, ни одного хода ±1%" if a.get("minutes")
                                           else "событий нет"))
            continue
        lines.append(f"\n{name}: минут {a['minutes']}, из них с ходом {a['events']} "
                     f"({a['move_share'] * 100:.0f}%), вверх {a['up_share'] * 100:.0f}%")
        rows = sorted(a["rows"], key=lambda x: -(x["low"] or -1))
        lines.append(f"  {'правило':<14}{'срез':<7}{'ходов':>8}{'эпиз.':>7}{'точность':>10}{'ниж.гр.':>9}"
                     f"{'база':>8}{'окуп. от':>10}  итог")
        for x in rows[:6]:
            be = "никогда" if x.get("break_even") is None else pct(x["break_even"])
            lines.append(f"  {x['rule']:<14}{x['cut']:<7}{x['events']:>8}{x['episodes']:>7}"
                         f"{pct(x['accuracy']):>10}{pct(x['low']):>9}{pct(x['base']):>8}{be:>10}  {x['verdict']}")
    lines.append("")
    if r["verdict"] == "direction_signal":
        lines.append("ИТОГ: direction_signal — направление видно: " + ", ".join(r["signals"]))
        lines.append("  окупается после издержек (оценка): " + (", ".join(r["tradeable"]) or
                     "нет — точность ниже порога, который задают издержки и доля минут без хода"))
    elif r["verdict"] == "no_direction":
        lines.append("ИТОГ: no_direction — ни стакан, ни поток не говорят, куда пойдёт ход")
    else:
        lines.append("ИТОГ: underpowered — мало независимых эпизодов")
    return "\n".join(lines)


def save_report(logs_root: Path, report: dict[str, Any]) -> Path:
    path = Path(logs_root) / "direction_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    return path

