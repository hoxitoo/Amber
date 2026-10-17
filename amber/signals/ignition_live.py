"""Live ignition warnings: "this coin is calm now; a sharp move is likely".

The ignition check (amber/backtest/ignition.py) found, on 30 days of data, that
ranked WITHIN each coin the model's score precedes a 1% move from a calm
market ~5.7x more often than chance (m30, lower bound 1.39 at z 3.79,
2026-10-09). That was one run. This module runs the same question live, as a
shadow channel, so it is confirmed or refuted on data the check never saw:

- `train_ignition`: once a day, the m30 arm of the check on the last 30 days.
  Earliest 85% trains the head, latest 15% calibrates it (Platt) and sets each
  coin's own alert threshold: the score its top `budget` share of calm bars
  reaches. A per-coin threshold is what "ranked within the coin" means live.
- `IgnitionScorer.score_universe`: every scan, each coin's newest bar is
  judged calm or not with the SAME `calm_now` and `feature_vector` the check
  uses, then scored. Every calm bar is recorded, alerting or not, so the
  forward ledger can measure the base rate it is compared with.

Nothing here is sent to Telegram unless `ignition.telegram` is true. It was
kept false until the forward ledger confirmed the signal (2026-10-17: lift
4.29, lower bound 2.14, 133 episodes over 9 days), and is true since.

Confirmation says warnings precede moves, not that they pay. Every shown
warning is therefore also traded, on paper, by one breakout bracket fixed
before any of its outcomes existed (`BRACKET_TRIGGER`, ledger.bracket_trade);
`summarize_bracket` gives its money verdict.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import time
from typing import Any

from amber.backtest.ignition import (
    _HOUR_CONTEXT,
    DEFAULT_ARMS,
    FEATURES,
    _hour_features,
    _Rolling,
    _scores,
    calm_now,
    candle_range_20,
    collect_arms,
    feature_vector,
    lead_from_rows,
)

logger = logging.getLogger(__name__)

SPEC = DEFAULT_ARMS[0]  # m30: calm 30 bars < 0.5% -> a 1% move within 30 bars
ARTIFACT_DIR = "ignition"
RECORDS_FILE = "ignition_calm.jsonl"
LEDGER_FILE = "ignition_ledger.jsonl"
LEDGER_STATE_KEY = "ignition_ledger_offsets"
SEEN_STATE_KEY = "ignition_seen"
NOTIFY_STATE_KEY = "ignition_notified"
KEEP_ARTIFACTS = 3
MIN_COIN_CALIB_ROWS = 100
MIN_TRAIN_DAYS = 7
MIN_TRAIN_POSITIVES = 30
TAIL_ROWS = _HOUR_CONTEXT + SPEC.window + 10
# Breakout bracket traded on every shown warning (ledger.bracket_trade): stops
# 0.3% either side of the price, target the warned-of 1% from it, stop back at
# it. Fixed 2026-10-17, before any bracket outcome existed. 0.3% sits outside
# what calm allows in the last 5 bars (0.1%) and inside the 30-bar calm range
# (0.5%), so a fill means the calm has broken.
BRACKET_TRIGGER = 0.003
# Warnings resolved before the bracket existed, re-scored once from the
# candles on disk by scripts/backfill_ignition_bracket.py.
BRACKET_BACKFILL_FILE = "ignition_bracket_backfill.jsonl"

# Plain-language names for the factors shown with an alert.
FACTOR_NAMES_RU = {
    "range_atr_14": "размах свечей последние 14 мин",
    "hr_range_240": "размах цены за 4 ч",
    "hr_compress_240_1440": "сжатие: 4 ч против суток",
    "hr_absret_240": "сдвиг цены за 4 ч",
    "hr_notional_60_1440": "объём за час против обычного",
    "bb_width_20": "ширина полос Боллинджера",
    "spread_bps": "спред",
    "btc_range_atr_14": "размах свечей BTC",
    "btc_absret_1": "BTC за 1 мин",
    "btc_absret_5": "BTC за 5 мин",
    "eth_absret_5": "ETH за 5 мин",
}


def _artifact_root(models_root: Path) -> Path:
    return Path(models_root) / ARTIFACT_DIR


def latest_artifact_path(models_root: Path) -> Path | None:
    root = _artifact_root(models_root)
    if not root.exists():
        return None
    files = sorted(root.glob("ignition_*.json"))
    return files[-1] if files else None


def artifact_age_min(models_root: Path) -> float | None:
    path = latest_artifact_path(models_root)
    if path is None:
        return None
    return (time.time() - path.stat().st_mtime) / 60.0


def train_ignition(
    features_root: Path,
    models_root: Path,
    *,
    days: float = 30.0,
    min_warmup_bars: int = 60,
    budget: float = 0.01,
    calib_frac: float = 0.15,
) -> dict[str, Any]:
    """Fit, calibrate and publish the ignition head. Returns a summary."""
    import numpy as np

    from amber.common.manifest import new_run_id
    from amber.models.recalibrate import _fit_platt
    from amber.models.train import _fit_head

    arms, n_symbols, names, _spreads = collect_arms(
        Path(features_root), specs=[SPEC], max_candles_per_symbol=int(days * 1440),
        min_warmup_bars=min_warmup_bars,
    )
    arm = arms[0]
    span_days = ((max(arm.ts) - min(arm.ts)) / 86_400_000) if len(arm.ts) else 0.0
    positives = sum(arm.y)
    if not n_symbols or span_days < MIN_TRAIN_DAYS or positives < MIN_TRAIN_POSITIVES:
        # A model trained on a day or two of calm history would publish
        # thresholds and probabilities that mean nothing.
        return {"status": "not_enough_data", "rows": len(arm.y), "days": round(span_days, 1),
                "positives": positives}
    x = np.frombuffer(arm.x, dtype=np.float32).reshape(-1, len(FEATURES)).astype(np.float64)
    y = np.frombuffer(arm.y, dtype=np.int8)
    ts = np.frombuffer(arm.ts, dtype=np.int64)
    sym = np.frombuffer(arm.sym, dtype=np.int16)
    order = np.argsort(ts, kind="stable")
    x, y, ts, sym = x[order], y[order], ts[order], sym[order]

    uniq = np.unique(ts)
    cut = uniq[int(len(uniq) * (1.0 - calib_frac))]
    gap_ms = (SPEC.horizon + 1) * 60_000  # labels reach horizon+1 bars ahead
    fit = ts < cut - gap_ms
    cal = ts >= cut
    if len(set(y[fit].tolist())) < 2 or len(set(y[cal].tolist())) < 2:
        return {"status": "single_class", "rows": int(len(y))}

    head = _fit_head(x[fit], y[fit].tolist())
    if head.get("type") == "constant":
        return {"status": "degenerate_model", "rows": int(len(y))}
    raw_cal = _scores(head, x[cal])
    platt = _fit_platt([float(v) for v in raw_cal], y[cal].tolist()) or {"method": "identity"}

    thresholds: dict[str, float] = {}
    coin_base: dict[str, float] = {}
    for idx, name in enumerate(names):
        in_coin = sym == idx
        if in_coin.any():
            coin_base[name] = float(y[in_coin].mean())
        cal_scores = raw_cal[sym[cal] == idx]
        if len(cal_scores) >= MIN_COIN_CALIB_ROWS:
            thresholds[name] = float(np.quantile(cal_scores, 1.0 - budget))

    run_id = new_run_id(prefix="ignition")
    artifact = {
        "run_id": run_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "spec": {"name": SPEC.name, "window": SPEC.window, "horizon": SPEC.horizon,
                 "calm_pct": SPEC.calm_pct, "barrier": SPEC.barrier},
        "features": list(FEATURES),
        "head": head,
        "calibration": platt,
        "thresholds": thresholds,
        "coin_base": coin_base,
        "base_rate": float(y.mean()),
        "budget": budget,
        "days": days,
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "fit_rows": int(fit.sum()),
        "calib_rows": int(cal.sum()),
        "min_warmup_bars": min_warmup_bars,
    }
    root = _artifact_root(models_root)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f"{run_id}.json.tmp"
    tmp.write_text(json.dumps(artifact), encoding="utf-8")
    tmp.replace(root / f"{run_id}.json")
    for old in sorted(root.glob("ignition_*.json"))[:-KEEP_ARTIFACTS]:
        old.unlink(missing_ok=True)
    logger.info(
        "ignition model %s: %s calm rows, %s moves, base %.4f, thresholds for %s/%s coins",
        run_id, artifact["rows"], artifact["positives"], artifact["base_rate"], len(thresholds), n_symbols,
    )
    return {"status": "ok", "run_id": run_id, "rows": artifact["rows"], "positives": artifact["positives"],
            "coins_with_threshold": len(thresholds), "symbols": n_symbols}


def live_vector(
    rows: list[dict[str, Any]],
    btc: dict[int, tuple[float, float, float]],
    eth: dict[int, tuple[float, float, float]],
    *,
    window: int,
    calm_pct: float,
    min_warmup_bars: int,
) -> list[float] | None:
    """Feature vector of the newest bar if it is calm, else None.

    Built from the same `calm_now`, `_hour_features` and `feature_vector` the
    training set is built from; a test pins the two to identical vectors on
    the same bars.
    """
    if len(rows) < window + 1:
        return None
    i = len(rows) - 1
    last = rows[i]
    if last.get("is_synthetic", False) or int(last.get("obs", 0) or 0) < min_warmup_bars:
        return None
    prices = [float(r.get("mid_price", 0.0) or 0.0) for r in rows]
    prefix = [0]
    for r in rows:
        prefix.append(prefix[-1] + int(bool(r.get("is_synthetic", False))))
    ranges = [candle_range_20(r) for r in rows]
    roll = _Rolling(prices)
    if not calm_now(roll, prefix, ranges, i, window=window, calm_pct=calm_pct):
        return None
    hours = _hour_features(prices, [float(r.get("notional_volume_1m", 0.0) or 0.0) for r in rows], roll)
    return feature_vector(last, btc, eth, hours[i])


class IgnitionScorer:
    """Scores each coin's newest bar with the latest published artifact."""

    def __init__(self, artifact: dict[str, Any], path: Path | None = None) -> None:
        self.art = artifact
        self.path = path
        self.spec = artifact["spec"]
        self._booster = None
        if artifact["head"].get("type") == "lightgbm":
            import lightgbm as lgb

            self._booster = lgb.Booster(model_str=artifact["head"]["booster"])

    @classmethod
    def load_latest(cls, models_root: Path) -> IgnitionScorer | None:
        path = latest_artifact_path(models_root)
        if path is None:
            return None
        try:
            return cls(json.loads(path.read_text(encoding="utf-8")), path)
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("ignition artifact unreadable (%s): %s", path, exc)
            return None

    def _raw(self, x: list[float]) -> float:
        import numpy as np

        if self._booster is not None:
            return float(self._booster.predict(np.asarray([x], dtype=float))[0])
        return float(_scores(self.art["head"], np.asarray([x], dtype=float))[0])

    def _factors(self, x: list[float], top: int = 3) -> list[str]:
        if self._booster is None:
            return []
        import numpy as np

        contrib = self._booster.predict(np.asarray([x], dtype=float), pred_contrib=True)[0][:-1]
        order = sorted(range(len(contrib)), key=lambda j: contrib[j], reverse=True)
        return [FACTOR_NAMES_RU.get(FEATURES[j], FEATURES[j]) for j in order[:top] if contrib[j] > 0]

    def score_rows(
        self,
        rows_by_symbol: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """One record per coin whose newest bar is calm."""
        from amber.signals.scorer import calibrated_prob

        btc = lead_from_rows(rows_by_symbol.get("BTCUSDT", []))
        eth = lead_from_rows(rows_by_symbol.get("ETHUSDT", []))
        window, calm_pct = int(self.spec["window"]), float(self.spec["calm_pct"])
        min_warmup = int(self.art.get("min_warmup_bars", 60))
        out: list[dict[str, Any]] = []
        for symbol, rows in sorted(rows_by_symbol.items()):
            ts = int(rows[-1].get("ts", 0) or 0) if rows else 0
            if (btc and ts not in btc) or (eth and ts not in eth):
                # The pipeline writes features symbol by symbol, so a scan can
                # land after this coin's new minute and before BTC's. The lead
                # features would silently read 0 ("BTC did not move") where
                # training always had the real values. Not marked seen, so
                # the next scan scores it once BTC/ETH have caught up.
                continue
            x = live_vector(rows, btc, eth, window=window, calm_pct=calm_pct, min_warmup_bars=min_warmup)
            if x is None:
                continue
            last = rows[-1]
            raw = self._raw(x)
            if not math.isfinite(raw):
                continue
            thr = self.art["thresholds"].get(symbol)
            alert = thr is not None and raw >= thr
            rec = {
                "event_ts": int(last.get("ts", 0) or 0),
                "symbol": symbol,
                "score": raw,
                "prob": calibrated_prob(raw, self.art.get("calibration", {"method": "identity"})),
                "threshold": thr,
                "coin_base": self.art["coin_base"].get(symbol),
                "alert": int(alert),
                "horizon_min": int(self.spec["horizon"]),
                "target_up_pct": float(self.spec["barrier"]),
                "model_run_id": self.art.get("run_id"),
                "price": float(last.get("mid_price", 0.0) or 0.0),
            }
            if alert:
                rec["factors"] = self._factors(x)
            out.append(rec)
        return out


def read_tails(features_root: Path, rows: int = TAIL_ROWS) -> dict[str, list[dict[str, Any]]]:
    from amber.common.jsonl import read_tail

    out: dict[str, list[dict[str, Any]]] = {}
    features_dir = Path(features_root) / "features"
    if not features_dir.exists():
        return out
    for symbol_dir in sorted(p for p in features_dir.iterdir() if p.is_dir()):
        parts = sorted(symbol_dir.glob("part-*.jsonl"))
        tail: list[dict[str, Any]] = []
        for part in reversed(parts):
            tail = read_tail(part, rows - len(tail)) + tail
            if len(tail) >= rows:
                break
        tail.sort(key=lambda r: int(r.get("ts", 0) or 0))
        if tail:
            out[symbol_dir.name] = tail
    return out


def scan_ignition(
    features_root: Path,
    models_root: Path,
    logs_root: Path,
    state: Any,
    *,
    scorer: IgnitionScorer | None = None,
    now_ms: int | None = None,
) -> list[dict[str, Any]]:
    """Score the universe once; append new calm bars to the records file.

    One record per coin per candle: the scan runs every minute and the newest
    feature row does not always advance, and a duplicate would be counted
    twice by the forward ledger.
    """
    scorer = scorer or IgnitionScorer.load_latest(models_root)
    if scorer is None:
        return []
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    seen = {k: int(v) for k, v in dict(state.get(SEEN_STATE_KEY) or {}).items()}
    notified = {k: int(v) for k, v in dict(state.get(NOTIFY_STATE_KEY) or {}).items()}
    fresh = []
    for rec in scorer.score_rows(read_tails(features_root)):
        if rec["event_ts"] <= seen.get(rec["symbol"], -1):
            continue
        seen[rec["symbol"]] = rec["event_ts"]
        rec["emitted_ms"] = now_ms
        # `alert` is the threshold crossing the check validated, kept for the
        # ledger. `notify` is what a person is shown: at most one per coin per
        # horizon, since a calm coin can sit over its threshold for minutes.
        cooldown_ms = rec["horizon_min"] * 60_000
        rec["notify"] = int(bool(rec["alert"]) and rec["event_ts"] - notified.get(rec["symbol"], -cooldown_ms) >= cooldown_ms)
        if rec["notify"]:
            notified[rec["symbol"]] = rec["event_ts"]
        fresh.append(rec)
    if fresh:
        logs_root.mkdir(parents=True, exist_ok=True)
        with (Path(logs_root) / RECORDS_FILE).open("a", encoding="utf-8") as fh:
            for rec in fresh:
                fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    state.set(SEEN_STATE_KEY, seen)
    state.set(NOTIFY_STATE_KEY, notified)
    return fresh


def alert_text(rec: dict[str, Any]) -> str:
    """The warning as the owner described it: calm now, P of a sharp move, why."""
    base = rec.get("coin_base")
    base_txt = f" (обычно {base * 100:.1f}%)" if base else ""
    factors = ", ".join(rec.get("factors") or []) or "—"
    return (
        f"{rec['symbol']}: сейчас спокойно. Вероятность хода ≥{rec['target_up_pct'] * 100:.0f}% в любую сторону "
        f"за {rec['horizon_min']} мин — {rec['prob'] * 100:.1f}%{base_txt}. Направление не прогнозируется. "
        f"Факторы: {factors}.{_bracket_text(rec)}"
    )


def _bracket_text(rec: dict[str, Any]) -> str:
    """The levels of the bracket the ledger books, so a person can place it.

    Records written before 2026-10-17 carry no price; their text has no levels.
    """
    price = float(rec.get("price", 0.0) or 0.0)
    if price <= 0:
        return ""

    def fmt(v: float) -> str:
        return f"{v:.{max(2, 5 - int(math.floor(math.log10(v))))}f}" if v > 0 else "-"

    t, b = BRACKET_TRIGGER, float(rec.get("target_up_pct", 0.01))
    return (
        f"\nЕсли торговать брекет (так его считает журнал): цена {fmt(price)}; "
        f"покупка-стоп {fmt(price * (1 + t))}, продажа-стоп {fmt(price * (1 - t))}; "
        f"сработал один — второй отменить; цель {fmt(price * (1 + b))} / {fmt(price * (1 - b))}, "
        f"стоп — обратно на {fmt(price)}; не сработал за {rec['horizon_min']} мин — отменить оба. "
        "Прибыльность брекета ещё проверяется."
    )


# Forward confirmation (CLAUDE.md section 11). Fixed 2026-10-09, before any
# live ignition record existed.
CONFIRM_MIN_DAYS = 7
CONFIRM_MIN_MOVE_EPISODES = 20


def summarize_ignition(logs_root: Path, *, since_ms: int | None = None) -> dict[str, Any]:
    """Do alerts precede moves more often than calm bars of the same coins?

    Every scored calm bar is in the ledger, so each coin's forward base rate
    is measured, not assumed. Alerts are compared with the hits expected if
    they had been ordinary calm bars of the same coins — coin choice cannot
    count. One pre-registered comparison, one-sided 95%.
    """
    from amber.backtest.label_sweep import _episodes, clustered_low, family_z

    path = Path(logs_root) / LEDGER_FILE
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") != "ok":
                    continue
                if since_ms is not None and int(r.get("event_ts", 0)) < since_ms:
                    continue
                rows.append(r)

    by_coin: dict[str, list[int]] = {}
    for r in rows:
        by_coin.setdefault(r["symbol"], []).append(int(r.get("move_hit", 0)))
    base = {c: sum(v) / len(v) for c, v in by_coin.items() if v}
    alerts = [r for r in rows if int(r.get("alert", 0))]
    horizon = int(SPEC.horizon)
    move_eps = _episodes([int(r["event_ts"]) for r in rows if int(r.get("move_hit", 0))], horizon)
    days = len({int(r["event_ts"]) // 86_400_000 for r in rows})
    out: dict[str, Any] = {
        "calm_bars": len(rows),
        "alerts": len(alerts),
        "notified": sum(int(r.get("notify", 0)) for r in rows),
        "days": days,
        "move_episodes": move_eps,
        "base_rate": (sum(int(r.get("move_hit", 0)) for r in rows) / len(rows)) if rows else None,
        "min_days": CONFIRM_MIN_DAYS,
        "min_move_episodes": CONFIRM_MIN_MOVE_EPISODES,
    }
    if not alerts:
        out.update({"verdict": "no_alerts_yet"})
        return out
    hits = sum(int(r.get("move_hit", 0)) for r in alerts)
    expected = sum(base[r["symbol"]] for r in alerts)
    precision = hits / len(alerts)
    expected_precision = expected / len(alerts)
    eps = _episodes([int(r["event_ts"]) for r in alerts], horizon)
    low = clustered_low(precision, eps, family_z(1))
    out.update({
        "hits": hits,
        "precision": precision,
        "expected_precision": expected_precision,
        "lift": (precision / expected_precision) if expected_precision > 0 else None,
        "lift_low": (low / expected_precision) if expected_precision > 0 else None,
        "alert_episodes": eps,
    })
    if days < CONFIRM_MIN_DAYS or move_eps < CONFIRM_MIN_MOVE_EPISODES:
        out["verdict"] = "underpowered"
    elif out["lift_low"] is not None and out["lift_low"] > 1.0:
        out["verdict"] = "confirmed"
    else:
        out["verdict"] = "not_confirmed"
    out["bracket"] = summarize_bracket(rows, _read_backfill(Path(logs_root)), horizon)
    return out


def _read_backfill(logs_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    path = logs_root / BRACKET_BACKFILL_FILE
    out: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
                out[(str(r["symbol"]), int(r["event_ts"]))] = r
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return out


def summarize_bracket(
    rows: list[dict[str, Any]],
    backfill: dict[tuple[str, int], dict[str, Any]],
    horizon: int,
) -> dict[str, Any]:
    """Does trading the bracket on every shown warning pay, after costs?

    One pre-registered rule, so one comparison: the episode-clustered mean net
    result per trade must have a one-sided 95% lower bound above 0, over at
    least `ledger.MIN_EPISODES` episodes and `ledger.MIN_DAYS` days — the same
    floors the main ledger's money verdict uses. Unfilled brackets are not
    trades and are counted separately.
    """
    from amber.backtest.label_sweep import family_z
    from amber.monitoring.ledger import BRACKET_COST, DAY_MS, MIN_DAYS, MIN_EPISODES, _episode_means, _mean_low

    shown = []
    for r in rows:
        if not int(r.get("notify", 0)):
            continue
        if "bracket_net" not in r:
            b = backfill.get((str(r["symbol"]), int(r["event_ts"])))
            if b is None:
                continue  # resolved before the bracket existed, not yet backfilled
            r = {**r, "bracket_side": b.get("bracket_side"), "bracket_net": b.get("bracket_net")}
        shown.append(r)
    traded = [r for r in shown if r.get("bracket_net") is not None]
    nets = [float(r["bracket_net"]) for r in traded]
    ep_means = _episode_means(traded, "bracket_net", horizon)
    low = _mean_low(ep_means, family_z(1))
    days = len({int(r["event_ts"]) // DAY_MS for r in traded})
    if len(ep_means) < MIN_EPISODES or days < MIN_DAYS:
        verdict = "underpowered"
    elif low is not None and low > 0:
        verdict = "profitable"
    else:
        verdict = "not_shown_profitable"
    return {
        "trigger": BRACKET_TRIGGER,
        "cost": BRACKET_COST,
        "shown_scored": len(shown),
        "filled": len(traded),
        "wins": sum(1 for v in nets if v > 0),
        "mean_net": (sum(nets) / len(nets)) if nets else None,
        "total_net": sum(nets),
        "mean_net_low": low,
        "episodes": len(ep_means),
        "days": days,
        "min_episodes": MIN_EPISODES,
        "min_days": MIN_DAYS,
        "verdict": verdict,
    }


SUMMARY_FILE = "ignition_summary.json"


def save_summary(logs_root: Path, summary: dict[str, Any]) -> None:
    """Published by the pipeline every 30 min. The ignition ledger grows by
    ~7,800 calm bars a day; re-reading it on every 15-second dashboard refresh
    is how the dashboard once took minutes to load."""
    logs_root = Path(logs_root)
    logs_root.mkdir(parents=True, exist_ok=True)
    tmp = logs_root / (SUMMARY_FILE + ".tmp")
    tmp.write_text(json.dumps({**summary, "computed_at": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
    tmp.replace(logs_root / SUMMARY_FILE)


def load_summary(logs_root: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((Path(logs_root) / SUMMARY_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def format_ignition_summary(s: dict[str, Any]) -> str:
    def pct(v: Any) -> str:
        return "-" if v is None else f"{v * 100:.2f}%"

    verdicts = {
        "no_alerts_yet": "предупреждений ещё не было",
        "underpowered": f"мало данных: нужно ≥{s['min_days']} дней и ≥{s['min_move_episodes']} независимых ходов",
        "confirmed": "ПОДТВЕРЖДЕНО: предупреждения опережают ходы чаще, чем обычная тишина тех же монет",
        "not_confirmed": "НЕ подтверждено: на новых данных предупреждения не лучше обычной тишины тех же монет",
    }
    lines = [
        "Предупреждения о зарождении — проверка вперёд",
        f"спокойных баров: {s['calm_bars']} · дней: {s['days']} · независимых ходов из тишины: {s['move_episodes']}",
        f"предупреждений (порог): {s['alerts']} · показано бы человеку (пауза 30 мин): {s['notified']}",
    ]
    if s.get("hits") is not None:
        lines.append(
            f"после предупреждения ход был в {pct(s['precision'])} случаев; ожидалось бы {pct(s['expected_precision'])} "
            f"для обычной тишины тех же монет · lift {s['lift']:.2f}, нижняя граница {s['lift_low']:.2f} "
            f"({s['alert_episodes']} эпизодов)"
        )
    lines.append(f"ИТОГ: {s['verdict']} — {verdicts.get(s['verdict'], '')}")
    b = s.get("bracket")
    if b:
        lines += ["", *format_bracket(b)]
    return "\n".join(lines)


def format_bracket(b: dict[str, Any]) -> list[str]:
    def bps(v: Any) -> str:
        return "-" if v is None else f"{v * 1e4:+.1f} bps"

    verdicts = {
        "underpowered": f"мало данных: нужно ≥{b['min_episodes']} эпизодов сделок за ≥{b['min_days']} дней",
        "profitable": "ПРИБЫЛЬНО: нижняя граница результата на сделку выше нуля после комиссий",
        "not_shown_profitable": "прибыльность НЕ показана: нижняя граница результата на сделку не выше нуля",
    }
    win = f"{b['wins'] / b['filled'] * 100:.0f}%" if b["filled"] else "-"
    return [
        f"Брекет на показанных предупреждениях (стопы ±{b['trigger'] * 100:.1f}%, цель 1%, стоп на цене "
        f"предупреждения, издержки {b.get('cost', 0) * 100:.2f}% на круг):",
        f"оценено показанных: {b['shown_scored']} · сработал брекет: {b['filled']} · выигрышных: {win} · "
        f"эпизодов: {b['episodes']} · дней: {b['days']}",
        f"результат на сделку: среднее {bps(b['mean_net'])}, нижняя граница {bps(b['mean_net_low'])} · "
        f"сумма {b['total_net'] * 100:+.2f}% (по 1 позиции на сделку)",
        f"ИТОГ брекета: {b['verdict']} — {verdicts.get(b['verdict'], '')}",
    ]
