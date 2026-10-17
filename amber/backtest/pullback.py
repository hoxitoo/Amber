"""Pullback to support on 4H bars: does buying the dip in a trend pay?

The owner brought this idea from a conversation about trading mechanics
(2026-10-16): in an uptrend, buy a pullback to support, stop under the zone,
target 3x the risk; mirrored with shorts in a downtrend. Fees matter far less
here than on 1m bars, because the targets are several percent.

Everything below was fixed on 2026-10-17, before the study saw real data,
and is not tuned afterwards: a rule chosen after looking would be fitted to
the history it is judged on.

Trend (decided on closed bars only): close > EMA200 and EMA50 > EMA200.

Two ways to find the support (`branch`):
- `drop`: price falls 3% below the highest high of the last 6 bars (24 h).
  The level is that -3% line.
- `retest`: a confirmed swing high (highest of 3 bars either side) is broken
  by a close; price later comes back to it within 60 bars. The zone is the
  level from -0.5 ATR to +0.25 ATR.

Three ways to enter (`entry`), from the conversation:
- `limit`: a resting buy at the level (zone top); maker fee.
- `confirm`: after price touches the level, the first bar within 3 that closes
  above it and above its own open; enter on that close, taker fee.
- `reclaim`: price pierces below the zone (drop: 1% below the level; retest:
  below the zone bottom), then within 2 bars closes back above the level;
  enter on that close, taker fee.

Stops: `limit` uses a fixed distance (drop: 1.5 ATR under the fill; retest:
0.25 ATR under the zone bottom); `confirm` / `reclaim` go 0.25 ATR under the
lowest low since the touch. Target = entry + 3 x risk. Out after 30 bars on
the close. Bars do not say what happened first inside them, so a bar touching
the stop is a stop even if it also touched the target, and the bar a limit
order fills on can stop it out but not take its profit.

Shorts are the same rules on the inverted price series (1/p: highs and lows
swap, a downtrend becomes an uptrend), mapped back to real prices for P&L.

Costs: Bybit base fees (taker 0.055%, maker 0.02%), 0.02% slippage on every
market exit, and the real funding paid or received while the trade is open.

Control: the same stop distance, target and costs, entered at random-like
bars (every 6th bar) in the same trend state. A support rule that only
matches its control is making its money from the trend, not from support.
"""

from __future__ import annotations

from typing import Any

TF_MS = 4 * 3_600_000
EMA_FAST, EMA_SLOW, ATR_N = 50, 200, 14
WARMUP = EMA_SLOW + 10
DROP_LOOKBACK, DROP_PCT, DROP_PIERCE = 6, 0.03, 0.01
PIVOT_K, RETEST_MIN_BARS, RETEST_WINDOW = 3, 2, 60
ZONE_UP_ATR, ZONE_DOWN_ATR, STOP_PAD_ATR, DROP_LIMIT_STOP_ATR = 0.25, 0.5, 0.25, 1.5
CONFIRM_BARS, RECLAIM_BARS = 3, 2
RR, TIMEOUT, CONTROL_EVERY = 3.0, 30, 6
TAKER, MAKER, SLIP = 0.00055, 0.0002, 0.0002
MIN_EPISODES = 30

BRANCHES = ("drop", "retest")
ENTRIES = ("limit", "confirm", "reclaim")
ARMS = tuple((b, e, s) for b in BRANCHES for e in ENTRIES for s in (1, -1))


def _ema(x: list[float], n: int) -> list[float]:
    a, out, v = 2.0 / (n + 1), [], x[0]
    for xi in x:
        v = a * xi + (1 - a) * v
        out.append(v)
    return out


def _atr(h: list[float], lo: list[float], c: list[float], n: int) -> list[float]:
    tr = [h[0] - lo[0]] + [max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])) for i in range(1, len(c))]
    out, s = [], 0.0
    for i, v in enumerate(tr):
        s += v - (tr[i - n] if i >= n else 0.0)
        out.append(s / min(i + 1, n))
    return out


def invert(bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """1/p: a short on p is a long on this series. Highs and lows swap."""
    return [{"ts": b["ts"], "open": 1 / b["open"], "high": 1 / b["low"], "low": 1 / b["high"],
             "close": 1 / b["close"]} for b in bars]


class _Series:
    def __init__(self, bars: list[dict[str, Any]]):
        self.ts = [int(b["ts"]) for b in bars]
        self.o = [float(b["open"]) for b in bars]
        self.h = [float(b["high"]) for b in bars]
        self.l = [float(b["low"]) for b in bars]
        self.c = [float(b["close"]) for b in bars]
        self.n = len(bars)
        self.ema_f, self.ema_s = _ema(self.c, EMA_FAST), _ema(self.c, EMA_SLOW)
        self.atr = _atr(self.h, self.l, self.c, ATR_N)

    def trend(self, t: int) -> bool:
        i = t - 1
        return self.c[i] > self.ema_s[i] and self.ema_f[i] > self.ema_s[i]

    def exit(self, bar: int, px: float, stop: float, target: float, intrabar: bool):
        """(exit_bar, exit_px, reason) or None when the window runs past the data."""
        last = bar + TIMEOUT
        if last >= self.n:
            return None
        for j in range(bar if intrabar else bar + 1, last + 1):
            if j == bar:  # the bar a limit order filled on: stop only
                if self.l[j] <= stop:
                    return j, stop * (1 - SLIP), "stop"
                continue
            if self.o[j] <= stop:
                return j, self.o[j] * (1 - SLIP), "stop"
            if self.l[j] <= stop:
                return j, stop * (1 - SLIP), "stop"
            if self.o[j] >= target:
                return j, self.o[j], "target"
            if self.h[j] >= target:
                return j, target, "target"
        return last, self.c[last] * (1 - SLIP), "timeout"


def _funding_sum(funding: list[tuple[int, float]], start: int, end: int) -> float:
    from bisect import bisect_right

    keys = [f[0] for f in funding]
    i, j = bisect_right(keys, start), bisect_right(keys, end)
    return sum(f[1] for f in funding[i:j])


def _book(s: _Series, side: int, symbol: str, funding, *, bar: int, px: float, stop: float,
          intrabar: bool, maker_entry: bool, arm: str) -> dict[str, Any] | None:
    if not (stop < px) or s.atr[bar - 1] <= 0:
        return None
    target = px + RR * (px - stop)
    res = s.exit(bar, px, stop, target, intrabar)
    if res is None:
        return None
    xbar, xpx, reason = res
    gross = (xpx / px - 1.0) if side > 0 else (1.0 - px / xpx)  # inverted series for shorts
    entry_ts = s.ts[bar] + (0 if intrabar else TF_MS)
    exit_ts = s.ts[xbar] + TF_MS
    fees = (MAKER if maker_entry else TAKER) + (MAKER if reason == "target" else TAKER)
    fund = _funding_sum(funding, entry_ts, exit_ts)
    return {"symbol": symbol, "arm": arm, "side": side, "entry_ts": entry_ts, "exit_ts": exit_ts,
            "bars": xbar - bar, "reason": reason, "gross": gross,
            "net": gross - fees - side * fund, "stop_atr": (px - stop) / s.atr[bar - 1]}


def arm_trades(bars, funding, branch: str, entry: str, side: int, symbol: str = "") -> list[dict[str, Any]]:
    """Every trade one arm takes on one symbol, oldest first, never overlapping."""
    s = _Series(bars if side > 0 else invert(bars))
    arm = f"{branch}/{entry}/{'long' if side > 0 else 'short'}"
    trades: list[dict[str, Any]] = []
    busy_until = -1
    state: dict[str, Any] | None = None
    resistances: list[float] = []
    flipped: dict[str, Any] | None = None

    for t in range(WARMUP, s.n):
        # Retest bookkeeping on closed bars, every bar, in or out of a trade.
        if branch == "retest":
            i = t - 1 - PIVOT_K
            if i >= PIVOT_K and s.h[i] >= max(s.h[i - PIVOT_K:i + PIVOT_K + 1]):
                resistances = (resistances + [s.h[i]])[-20:]
            broken = [r for r in resistances if s.c[t - 1] > r]
            if broken:
                flipped = {"level": max(broken), "break": t - 1}
                resistances = [r for r in resistances if r not in broken]
        if t <= busy_until:
            continue

        if state is None:
            atr = s.atr[t - 1]
            setup = None
            if atr > 0 and s.trend(t):
                if branch == "drop":
                    level = max(s.h[t - DROP_LOOKBACK:t]) * (1 - DROP_PCT)
                    if s.c[t - 1] > level:
                        setup = {"lim": level, "pierce": level * (1 - DROP_PIERCE), "reclaim": level,
                                 "limit_stop": None}
                elif flipped is not None:
                    lvl, age = flipped["level"], t - flipped["break"]
                    if RETEST_MIN_BARS < age <= RETEST_WINDOW and s.c[t - 1] > lvl:
                        setup = {"lim": lvl + ZONE_UP_ATR * atr, "pierce": lvl - ZONE_DOWN_ATR * atr,
                                 "reclaim": lvl, "limit_stop": lvl - (ZONE_DOWN_ATR + STOP_PAD_ATR) * atr}
                    elif age > RETEST_WINDOW:
                        flipped = None
            if setup is None or s.l[t] > setup["lim"]:
                continue
            if branch == "retest":
                flipped = None  # one touch per broken level
            if entry == "limit":
                px = min(s.o[t], setup["lim"])
                stop = setup["limit_stop"] if setup["limit_stop"] is not None else px - DROP_LIMIT_STOP_ATR * atr
                tr = _book(s, side, symbol, funding, bar=t, px=px, stop=stop, intrabar=True, maker_entry=True,
                           arm=arm)
                if tr:
                    trades.append(tr)
                    busy_until = t + tr["bars"]
                continue
            state = {**setup, "touch": t, "low": s.l[t], "pierced": None, "atr": atr}

        # Touched: wait for the confirmation or the reclaim.
        state["low"] = min(state["low"], s.l[t])
        if state["pierced"] is None and s.l[t] <= state["pierce"]:
            state["pierced"] = t
        go = False
        if entry == "confirm":
            go = s.c[t] > state["lim"] and s.c[t] > s.o[t]
            expired = t >= state["touch"] + CONFIRM_BARS
        else:
            p = state["pierced"]
            go = p is not None and s.c[t] > state["reclaim"] and t <= p + RECLAIM_BARS
            expired = (p is not None and t >= p + RECLAIM_BARS) or (p is None and t >= state["touch"] + CONFIRM_BARS)
        if go:
            stop = state["low"] - STOP_PAD_ATR * state["atr"]
            tr = _book(s, side, symbol, funding, bar=t, px=s.c[t], stop=stop, intrabar=False, maker_entry=False,
                       arm=arm)
            state = None
            if tr:
                trades.append(tr)
                busy_until = t + tr["bars"]
        elif expired:
            state = None
    return trades


def control_trades(bars, funding, side: int, stop_atr: float, symbol: str = "") -> list[dict[str, Any]]:
    """Same stop distance, target and costs; entered on every 6th trend bar."""
    s = _Series(bars if side > 0 else invert(bars))
    out = []
    for t in range(WARMUP, s.n):
        if t % CONTROL_EVERY or not s.trend(t):
            continue
        px = s.c[t]
        tr = _book(s, side, symbol, funding, bar=t, px=px, stop=px - stop_atr * s.atr[t - 1], intrabar=False,
                   maker_entry=False, arm="control")
        if tr:
            out.append(tr)
    return out


def _episode_means(trades: list[dict[str, Any]]) -> list[float]:
    """Trades entered within one timeout of each other, on any symbol, are one
    observation: BTC and ETH dip together."""
    groups: list[list[float]] = []
    anchor = None
    for tr in sorted(trades, key=lambda x: x["entry_ts"]):
        if anchor is None or tr["entry_ts"] - anchor > TIMEOUT * TF_MS:
            groups.append([])
            anchor = tr["entry_ts"]
        groups[-1].append(tr["net"])
    return [sum(g) / len(g) for g in groups]


def _mean(v: list[float]) -> float | None:
    return sum(v) / len(v) if v else None


def run_study(series: dict[str, tuple[list[dict[str, Any]], list[tuple[int, float]]]]) -> dict[str, Any]:
    from amber.backtest.label_sweep import family_z
    from amber.monitoring.ledger import _mean_low

    z = family_z(len(ARMS))
    arms_out = []
    for branch, entry, side in ARMS:
        trades: list[dict[str, Any]] = []
        for sym, (bars, funding) in series.items():
            trades += arm_trades(bars, funding, branch, entry, side, sym)
        stop_atrs = sorted(tr["stop_atr"] for tr in trades)
        stop_atr = stop_atrs[len(stop_atrs) // 2] if stop_atrs else DROP_LIMIT_STOP_ATR
        control: list[dict[str, Any]] = []
        for sym, (bars, funding) in series.items():
            control += control_trades(bars, funding, side, stop_atr, sym)
        eps = _episode_means(trades)
        low = _mean_low(eps, z)
        ctrl_mean = _mean([tr["net"] for tr in control])
        if len(eps) < MIN_EPISODES:
            verdict = "underpowered"
        elif low is not None and low > 0:
            verdict = "beats_trend" if ctrl_mean is not None and low > ctrl_mean else "matched_by_trend"
        else:
            verdict = "not_profitable"
        nets = [tr["net"] for tr in sorted(trades, key=lambda x: x["entry_ts"])]
        half = len(nets) // 2
        arms_out.append({
            "arm": f"{branch}/{entry}/{'long' if side > 0 else 'short'}",
            "trades": len(trades), "episodes": len(eps),
            "win_rate": (sum(1 for v in nets if v > 0) / len(nets)) if nets else None,
            "mean_net": _mean(nets), "mean_gross": _mean([tr["gross"] for tr in trades]),
            "mean_net_low": low, "control_mean": ctrl_mean, "control_trades": len(control),
            "first_half": _mean(nets[:half]), "second_half": _mean(nets[half:]),
            "stop_atr": stop_atr, "verdict": verdict,
        })
    span = {sym: (bars[0]["ts"], bars[-1]["ts"], bars[-1]["close"] / bars[WARMUP]["close"] - 1)
            for sym, (bars, _) in series.items() if len(bars) > WARMUP}
    return {"z": z, "arms": arms_out, "span": span, "min_episodes": MIN_EPISODES}


def format_study(r: dict[str, Any], source: str) -> str:
    from datetime import datetime, timezone

    def bps(v):
        return "-" if v is None else f"{v * 1e4:+.0f}"

    def day(ms):
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    lines = [f"Откаты к поддержке, 4H, источник: {source}. Границы с поправкой на {len(r['arms'])} вариантов "
             f"(z={r['z']:.2f}), вывод от {r['min_episodes']} эпизодов."]
    for sym, (a, b, bh) in r["span"].items():
        lines.append(f"  {sym}: {day(a)} — {day(b)}, купить и держать: {bh * 100:+.0f}%")
    lines += ["", f"{'вариант':<22}{'сделок':>7}{'эпиз.':>6}{'выигр.':>7}{'до ком.':>8}{'чистыми':>8}"
                  f"{'ниж.гр.':>8}{'контроль':>9}{'1-я пол.':>9}{'2-я пол.':>9}  итог",
              "  (все результаты — в bps на сделку)"]
    for a in r["arms"]:
        win = "-" if a["win_rate"] is None else f"{a['win_rate'] * 100:.0f}%"
        lines.append(f"{a['arm']:<22}{a['trades']:>7}{a['episodes']:>6}{win:>7}{bps(a['mean_gross']):>8}"
                     f"{bps(a['mean_net']):>8}{bps(a['mean_net_low']):>8}{bps(a['control_mean']):>9}"
                     f"{bps(a['first_half']):>9}{bps(a['second_half']):>9}  {a['verdict']}")
    wins = [a["arm"] for a in r["arms"] if a["verdict"] == "beats_trend"]
    trend = [a["arm"] for a in r["arms"] if a["verdict"] == "matched_by_trend"]
    lines.append("")
    if wins:
        lines.append("ИТОГ: прибыльно и лучше контроля: " + ", ".join(wins))
    elif trend:
        lines.append("ИТОГ: в плюсе, но не лучше случайного входа в том же тренде (деньги от тренда, не от "
                     "поддержки): " + ", ".join(trend))
    else:
        lines.append("ИТОГ: ни один вариант не показал прибыли после издержек с учётом поправки")
    lines.append("Итоги: beats_trend — плюс и лучше контроля; matched_by_trend — плюс, но не лучше контроля; "
                 "not_profitable — плюс не показан; underpowered — мало эпизодов.")
    return "\n".join(lines)
