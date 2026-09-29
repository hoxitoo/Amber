"""Forward ledger: every live alert, scored after its horizon, kept for good.

Every evaluation before this one lived in the test segment of a rolling
training window — 7 hours at a 48h window, 18 at 72h — so it could never
accumulate evidence: each retrain slid the segment forward and the previous
measurement was gone. The ledger is the opposite. It is out-of-sample by
construction (an alert is logged before its outcome exists), it only grows,
and it answers the question the owner is actually asking: would acting on
these alerts have made money?

Two sources are scored identically:

- `model`: what the scanner emitted (`signals.jsonl`).
- `rule`: a shadow scanner that fires on raw `range_atr_14` alone, at the
  alert rate the model runs at, through the same universe, spread filter and
  gate (`shadow_signals.jsonl`). The baseline check found the model
  indistinguishable from this rule; the ledger keeps asking, forward.

Each alert is traded by two rules fixed before any outcome was seen, because a
rule chosen after looking at the ledger would be fitted to it:

- `momentum`: the direction of the alert bar's close-to-close return.
- `fade`: the opposite.

Both use the alert's own barrier as take-profit and stop-loss, time out at the
horizon, and pay `cost` per round trip. Entry is the close of the first bar
that closes after the alert was actually sent (`emitted_ms`), never sooner
than `lag_bars` after the alert bar: the pipeline and the scanner each run once
a minute, so an alert can arrive two bars late, and entering at a price that
was already gone would flatter every rule. A bar touching both barriers is
booked as a loss: 1m bars do not say which came first.

Barriers are checked against candle highs and lows, because that is where a
resting take-profit or stop fills. The training label is not: it follows the
per-minute bid/ask mid, which has no wicks. So the ledger's move hit rate runs
above the model's calibrated P(move) and the two are not comparable; compare
model with rule inside the ledger, where both are scored the same way.

Amber places no orders. This is bookkeeping over public candles.
"""

from __future__ import annotations

from bisect import bisect_left
import json
import logging
import math
from pathlib import Path
import time
from typing import Any

from amber.backtest.label_sweep import _episodes, _wilson_low, family_z
from amber.monitoring.quality_report import _CandleIndex, _event_ts_ms

logger = logging.getLogger(__name__)

STEP_MS = 60_000
SOURCES = {"model": "signals.jsonl", "rule": "shadow_signals.jsonl"}
LEDGER_FILE = "ledger.jsonl"
STATE_KEY = "ledger_offsets"
RULES = ("momentum", "fade")

# Round trip, as the backtest books it: 5 bps slippage + 4 bps fee.
DEFAULT_COST = 0.0009
DEFAULT_LAG_BARS = 1
# An alert whose candles still have not arrived after this long is written off
# as `no_data` rather than blocking every later alert behind it.
DEFAULT_EXPIRE_HOURS = 6.0
# Below this many independent episodes a verdict is not issued (section 11 of
# CLAUDE.md). Higher than the analysis tools' 10 because this one decides money.
MIN_EPISODES = 30
# Ceiling on candles held per symbol per cycle: 3 days. Normally a cycle needs
# ~80 (alerts wait ~17 minutes); this only binds after the pipeline was down
# with alerts pending. Alerts older than it are written off as `no_data`, which
# keeps a long outage from loading weeks of candles for 27 symbols at once.
MAX_CANDLE_ROWS = 4320


def resolve_alert(
    candles: list[dict[str, Any]],
    ts_list: list[int],
    event_ts: int,
    *,
    horizon: int,
    barrier: float,
    lag_bars: int = DEFAULT_LAG_BARS,
    cost: float = DEFAULT_COST,
    emitted_ms: int | None = None,
) -> dict[str, Any] | None:
    """Score one alert against the candles that followed it.

    Returns None while the horizon has not elapsed in the data (try again next
    cycle), otherwise a record whose `status` is `ok`, `gap` (a missing or
    gap-filled bar inside the window: flat synthetic bars would read as "no
    move") or `no_bar` (the alert bar itself is not in the data).
    """
    if barrier <= 0 or horizon <= 0:
        return {"status": "bad_signal"}
    # Enter on the first bar that closes after the alert was sent, and never
    # sooner than `lag_bars`. The alert bar closes at event_ts + 1 bar; an alert
    # sent d ms after that can first be acted on at the close of bar
    # floor(d / bar) + 1. Signals logged before emission times were recorded
    # fall back to `lag_bars`.
    if emitted_ms is not None:
        delay = max(0, int(emitted_ms) - (event_ts + STEP_MS))
        lag_bars = max(lag_bars, delay // STEP_MS + 1)
    i0 = bisect_left(ts_list, event_ts)
    if i0 >= len(ts_list):
        return None  # the alert bar has not been normalised yet
    if ts_list[i0] != event_ts or i0 == 0:
        return {"status": "no_bar"}
    last_i = i0 + lag_bars + horizon
    if last_i >= len(ts_list):
        return None
    window = range(i0 - 1, last_i + 1)
    for k, i in enumerate(window):
        expected_ts = event_ts + (k - 1) * STEP_MS
        if ts_list[i] != expected_ts or bool(candles[i].get("is_synthetic", False)):
            return {"status": "gap"}

    prev_close = float(candles[i0 - 1].get("close", 0.0) or 0.0)
    alert_close = float(candles[i0].get("close", 0.0) or 0.0)
    entry = float(candles[i0 + lag_bars].get("close", 0.0) or 0.0)
    if prev_close <= 0 or alert_close <= 0 or entry <= 0:
        return {"status": "gap"}
    bar_dir = (alert_close > prev_close) - (alert_close < prev_close)

    up_level, down_level = entry * (1 + barrier), entry * (1 - barrier)
    first_touch: int | None = None  # +1 up, -1 down, 0 both inside one bar
    for i in range(i0 + lag_bars + 1, last_i + 1):
        c = candles[i]
        up = float(c.get("high", 0.0) or 0.0) >= up_level
        down = 0 < float(c.get("low", 0.0) or 0.0) <= down_level
        if up or down:
            first_touch = 0 if (up and down) else (1 if up else -1)
            break
    exit_close = float(candles[last_i].get("close", 0.0) or 0.0)

    def _trade(direction: int) -> float | None:
        if direction == 0:
            return None  # a flat alert bar has no direction to follow or fade
        if first_touch is None:
            gross = direction * (exit_close / entry - 1.0)
        elif first_touch == 0:
            gross = -barrier  # order within the bar unknown: assume the stop
        else:
            gross = barrier if first_touch == direction else -barrier
        return gross - cost

    return {
        "status": "ok",
        "move_hit": int(first_touch is not None),
        "first_touch": first_touch,
        "bar_dir": bar_dir,
        "entry": entry,
        "entry_lag": lag_bars,
        "momentum_net": _trade(bar_dir),
        "fade_net": _trade(-bar_dir),
    }


def _read_from(path: Path, offset: int) -> list[tuple[int, dict[str, Any] | None]]:
    """Complete lines after `offset`, each with the offset just past it. A
    trailing partial line (the scanner mid-write) is left for the next cycle."""
    out: list[tuple[int, dict[str, Any] | None]] = []
    if not path.exists():
        return out
    with path.open("rb") as fh:
        fh.seek(offset)
        pos = offset
        for raw in fh:
            if not raw.endswith(b"\n"):
                break
            pos += len(raw)
            try:
                row = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                row = None
            out.append((pos, row if isinstance(row, dict) else None))
    return out


def _alert_fields(source: str, row: dict[str, Any]) -> dict[str, Any] | None:
    event_ts = _event_ts_ms(row.get("event_ts"))
    symbol = str(row.get("symbol", "") or "")
    if event_ts is None or not symbol:
        return None
    fields = {
        "source": source,
        "symbol": symbol,
        "event_ts": event_ts,
        "horizon": int(row.get("horizon_min", 0) or 0),
        "barrier": float(row.get("target_up_pct", 0.0) or 0.0),
        # Which weights fired it. Absent on signals logged before 2026-09-29.
        "model_run_id": (row.get("market_context") or {}).get("model_run_id") or row.get("model_run_id"),
        "emitted_ms": (row.get("market_context") or {}).get("emitted_ms") or row.get("emitted_ms"),
    }
    if source == "model":
        fields["score"] = row.get("prob_move_calibrated")
    else:
        fields["score"] = row.get("range_atr_14")
    return fields


def update_ledger(
    logs_root: Path,
    raw_root: Path,
    state: Any,
    *,
    lag_bars: int = DEFAULT_LAG_BARS,
    cost: float = DEFAULT_COST,
    expire_hours: float = DEFAULT_EXPIRE_HOURS,
    now_ms: int | None = None,
) -> int:
    """Resolve every alert whose horizon has elapsed; append to the ledger.

    Progress is a byte offset per source, advanced only past alerts that are
    finished, so a restart never scores an alert twice and never skips one.
    Alerts are appended in time order, so the first one still waiting for
    candles holds back the rest — for at most `expire_hours`.
    """
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    try:
        offsets = {k: int(v) for k, v in dict(state.get(STATE_KEY) or {}).items()}
    except Exception:
        offsets = {}
    written = 0
    ledger_path = logs_root / LEDGER_FILE

    for source, name in SOURCES.items():
        path = logs_root / name
        if source not in offsets:
            # First run: start at the end. signals.jsonl already holds weeks
            # of alerts from other models, thresholds and training windows;
            # mixed into the ledger they would describe none of them.
            offsets[source] = path.stat().st_size if path.exists() else 0
            logger.info("ledger: starting %s at byte %s; earlier alerts are not scored", source, offsets[source])
        offset = offsets[source]
        if path.exists() and path.stat().st_size < offset:
            # The log was rotated or truncated under us. Starting over would
            # re-score everything; start from the end instead and say so.
            logger.warning("ledger: %s shrank below its offset; resuming from its end", name)
            offset = path.stat().st_size
        pending = _read_from(path, offset)
        if not pending:
            offsets[source] = offset
            continue

        alerts = [(pos, _alert_fields(source, row) if row else None) for pos, row in pending]
        oldest = min((a["event_ts"] for _, a in alerts if a), default=now_ms)
        rows_needed = min(MAX_CANDLE_ROWS, (now_ms - oldest) // STEP_MS + 64)
        index = _CandleIndex(raw_root, max_rows=int(rows_needed))

        records: list[dict[str, Any]] = []
        for pos, alert in alerts:
            if alert is None:
                offset = pos  # unreadable line: nothing to score, move past it
                continue
            if now_ms - alert["event_ts"] > (MAX_CANDLE_ROWS - 64) * STEP_MS:
                records.append({**alert, "status": "no_data", "lag_bars": lag_bars, "cost": cost,
                                "resolved_ms": now_ms})
                offset = pos
                continue
            candles = index.candles(alert["symbol"])
            res = resolve_alert(
                candles, index.timestamps(alert["symbol"]), alert["event_ts"],
                horizon=alert["horizon"], barrier=alert["barrier"], lag_bars=lag_bars, cost=cost,
                emitted_ms=alert["emitted_ms"],
            )
            if res is None:
                if now_ms - alert["event_ts"] < expire_hours * 3_600_000:
                    break  # not yet; everything after it is later still
                res = {"status": "no_data"}
            records.append({**alert, **res, "lag_bars": lag_bars, "cost": cost, "resolved_ms": now_ms})
            offset = pos

        if records:
            logs_root.mkdir(parents=True, exist_ok=True)
            with ledger_path.open("a", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            written += len(records)
        offsets[source] = offset

    state.set(STATE_KEY, offsets)
    return written


def _mean_low(values: list[float], z: float) -> float | None:
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m - z * math.sqrt(var / len(values))


def _episode_means(rows: list[dict[str, Any]], key: str, horizon: int) -> list[float]:
    """Mean trade result per independent episode — the unit the bound counts.

    Alerts within one horizon of each other, on any symbol, are one market
    event (label_sweep._episodes); treating them as separate trades would
    shrink the interval by the cluster size.
    """
    ordered = sorted((r for r in rows if r.get(key) is not None), key=lambda r: r["event_ts"])
    groups: list[list[float]] = []
    anchor: int | None = None
    span = max(1, horizon) * STEP_MS
    for r in ordered:
        if anchor is None or r["event_ts"] - anchor > span:
            groups.append([])
            anchor = r["event_ts"]
        groups[-1].append(float(r[key]))
    return [sum(g) / len(g) for g in groups]


def summarize_ledger(
    logs_root: Path,
    *,
    min_episodes: int = MIN_EPISODES,
    since_ms: int | None = None,
) -> dict[str, Any]:
    """Per source: move hit rate, and each trading rule's net result per trade.

    Every bound is episode-clustered and corrected for the whole family
    compared here (2 sources x (hit rate + 2 rules)), so picking the best line
    of the table does not by itself produce a winner.
    """
    path = logs_root / LEDGER_FILE
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if since_ms is not None and int(r.get("event_ts", 0)) < since_ms:
                    continue
                rows.append(r)

    z = family_z(len(SOURCES) * (1 + len(RULES)))
    out: dict[str, Any] = {"z": z, "min_episodes": min_episodes, "sources": {}}
    for source in SOURCES:
        mine = [r for r in rows if r.get("source") == source]
        ok = [r for r in mine if r.get("status") == "ok"]
        horizon = max((int(r.get("horizon", 15) or 15) for r in ok), default=15)
        eps = _episodes([r["event_ts"] for r in ok], horizon)
        hits = sum(int(r.get("move_hit", 0)) for r in ok)
        hit_rate = hits / len(ok) if ok else None
        entry: dict[str, Any] = {
            "alerts": len(mine),
            "scored": len(ok),
            "skipped": {s: sum(1 for r in mine if r.get("status") == s) for s in ("gap", "no_bar", "no_data")},
            "episodes": eps,
            "underpowered": eps < min_episodes,
            "move_hit_rate": hit_rate,
            "move_hit_rate_low": (
                _wilson_low(int(round(hit_rate * eps)), eps, z) if ok and eps else None
            ),
            "rules": {},
        }
        for rule in RULES:
            key = f"{rule}_net"
            traded = [float(r[key]) for r in ok if r.get(key) is not None]
            ep_means = _episode_means(ok, key, horizon)
            low = _mean_low(ep_means, z)
            mean = sum(traded) / len(traded) if traded else None
            if len(ep_means) < min_episodes:
                verdict = "underpowered"
            elif low is not None and low > 0:
                verdict = "profitable"
            else:
                verdict = "not_shown_profitable"
            entry["rules"][rule] = {
                "trades": len(traded),
                "episodes": len(ep_means),
                "win_rate": (sum(1 for v in traded if v > 0) / len(traded)) if traded else None,
                "mean_net": mean,
                "total_net": sum(traded) if traded else 0.0,
                "mean_net_low": low,
                "verdict": verdict,
            }
        out["sources"][source] = entry

    model, rule = out["sources"]["model"], out["sources"]["rule"]
    if model["underpowered"] or rule["underpowered"]:
        out["model_vs_rule"] = "underpowered"
    elif model["move_hit_rate_low"] is not None and rule["move_hit_rate"] is not None \
            and model["move_hit_rate_low"] > rule["move_hit_rate"]:
        out["model_vs_rule"] = "model_better"
    elif rule["move_hit_rate_low"] is not None and model["move_hit_rate"] is not None \
            and rule["move_hit_rate_low"] > model["move_hit_rate"]:
        out["model_vs_rule"] = "rule_better"
    else:
        out["model_vs_rule"] = "indistinguishable"
    return out


def format_summary(summary: dict[str, Any]) -> str:
    def pct(v: Any, digits: int = 1) -> str:
        return "-" if v is None else f"{v * 100:.{digits}f}%"

    def bps(v: Any) -> str:
        return "-" if v is None else f"{v * 1e4:+.1f}"

    lines = [
        f"Forward ledger — bounds at z={summary['z']:.2f}, verdicts need "
        f">= {summary['min_episodes']} episodes",
        "",
        f"{'source':<7} {'alerts':>6} {'scored':>6} {'episodes':>8} {'move hit':>9} {'(low)':>7}",
    ]
    for source, s in summary["sources"].items():
        lines.append(
            f"{source:<7} {s['alerts']:>6} {s['scored']:>6} {s['episodes']:>8} "
            f"{pct(s['move_hit_rate']):>9} {pct(s['move_hit_rate_low']):>7}"
        )
    lines += [
        "",
        f"{'source':<7} {'rule':<9} {'trades':>6} {'win':>6} {'mean bps':>9} "
        f"{'low bps':>8} {'total %':>8}  verdict",
    ]
    for source, s in summary["sources"].items():
        for rule, r in s["rules"].items():
            lines.append(
                f"{source:<7} {rule:<9} {r['trades']:>6} {pct(r['win_rate'], 0):>6} "
                f"{bps(r['mean_net']):>9} {bps(r['mean_net_low']):>8} "
                f"{r['total_net'] * 100:>+8.2f}  {r['verdict']}"
            )
    lines += ["", f"model vs rule (move hit rate): {summary['model_vs_rule']}"]
    return "\n".join(lines)
