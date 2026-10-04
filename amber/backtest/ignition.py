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
bars is under `calm_pct` and over the last 5 bars under a fifth of that — so "already moving" is excluded by construction and
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
- `no_precursor`: nothing beats chance — on these inputs a quiet market gives
  no warning, and earlier warning needs earlier data (roadmap D2: order book
  and tick-level trade flow).
- `underpowered`: fewer than MIN_EPISODES independent episodes.

Read-only: trains in memory, writes only logs/ignition_check.json.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from amber.backtest.label_sweep import _precision_at_budget, _Series, family_z, load_series
from amber.labeling.events import label_path
from amber.models.features import MODEL_FEATURES
from amber.models.split import make_holdout_splits

logger = logging.getLogger(__name__)

REPORT_FILE = "ignition_check.json"
MIN_EPISODES = 10
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
DEFAULT_BARRIER = 0.010
# 1% of calm test rows. Calm rows are a subset, so this is fewer alerts than
# the live scanner fires.
DEFAULT_BUDGET = 0.01

_VERDICT_RANK = {
    "precursor_found": 5,
    "signal_matched_by": 4,
    "single_feature_signal": 3,
    "no_precursor": 2,
    "underpowered": 1,
}


def build_ignition_rows(
    series_by_symbol: dict[str, _Series],
    *,
    window: int,
    horizon: int,
    calm_pct: float = DEFAULT_CALM_PCT,
    barrier: float = DEFAULT_BARRIER,
) -> tuple[list[dict[str, Any]], int]:
    """Calm rows labelled with whether price then travels `barrier`.

    Returns (rows, clean_rows_considered). A gap-filled bar anywhere in the
    lookback or the forward window disqualifies the row: synthetic bars are
    flat, so a gap would read as calm and then as "no move".
    """
    out: list[dict[str, Any]] = []
    considered = 0
    for symbol, series in series_by_symbol.items():
        prices, rows = series.prices, series.rows
        n = len(prices)
        synthetic = [bool(r.get("is_synthetic", False)) for r in rows]
        for i in series.clean_idx:
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
            labels = label_path(prices[i + 1 : hi + 1], up_pct=barrier, down_pct=barrier, shape="one_sided")
            src = rows[i]
            row = {name: src.get(name, 0.0) for name in MODEL_FEATURES}
            row.update({
                "symbol": symbol,
                "ts": int(src.get("ts", 0) or 0),
                "up_hit": int(labels["up_hit"]),
                "down_hit": int(labels["down_hit"]),
                "move_hit": int(bool(labels["up_hit"] or labels["down_hit"])),
                "first_hit": int(labels["first_hit"] or 0),
                "horizon_steps": horizon,
                "up_pct": barrier,
                "down_pct": barrier,
            })
            out.append(row)
    out.sort(key=lambda r: (r["ts"], r["symbol"]))
    return out, considered


def _best_rule(
    test: list[dict[str, Any]], y: list[int], ts: list[int], *, budget: float, z: float, horizon: int
) -> dict[str, Any]:
    """Every feature as a one-line rule, both directions, best achieved lift.

    Chosen on the test segment itself, which flatters the rule — deliberately:
    the model has to beat the luckiest single feature, not a fair one.
    """
    best: dict[str, Any] | None = None
    for name in MODEL_FEATURES:
        values = [float(r.get(name, 0.0) or 0.0) for r in test]
        if len(set(values)) < 2:
            continue
        for sign in (1.0, -1.0):
            res = _precision_at_budget([sign * v for v in values], y, ts, budget, z, horizon)
            lift = res.get("lift") or 0.0
            if best is None or lift > (best.get("lift") or 0.0):
                best = {**res, "rule": f"{'+' if sign > 0 else '-'}{name}"}
    return best or {"rule": None, "lift": None, "lift_ci_low_clustered": None, "episodes": 0}


def _verdict(model: dict[str, Any], rule: dict[str, Any]) -> str:
    if (model.get("episodes") or 0) < MIN_EPISODES:
        return "underpowered"
    m_lo = model.get("lift_ci_low_clustered") or 0.0
    r_lift = rule.get("lift") or 0.0
    r_lo = rule.get("lift_ci_low_clustered") or 0.0
    if m_lo > 1.0 and m_lo > r_lift:
        return "precursor_found"
    if m_lo > 1.0:
        return f"signal_matched_by:{rule.get('rule')}"
    if r_lo > 1.0 and (rule.get("episodes") or 0) >= MIN_EPISODES:
        return f"single_feature_signal:{rule.get('rule')}"
    return "no_precursor"


def evaluate_ignition_arm(
    rows: list[dict[str, Any]],
    *,
    window: int,
    horizon: int,
    budget: float = DEFAULT_BUDGET,
    train_frac: float = 0.6,
    calib_frac: float = 0.15,
    z: float = 1.96,
    importance: bool = True,
) -> dict[str, Any]:
    from amber.models.dataset_io import order_with_pseudo_time, split_rows
    from amber.models.train import _fit_dual, _predict_head

    if not rows:
        return {"status": "no_calm_rows"}
    ordered, pts, _mode = order_with_pseudo_time(rows)
    # The label reaches horizon+1 bars past the row; the purge gap must cover it.
    splits = make_holdout_splits(pts, train_frac=train_frac, calib_frac=calib_frac, gap=max(30, horizon + 1))
    if splits is None:
        return {"status": "too_small", "rows": len(rows)}
    seg = split_rows(ordered, pts, splits)
    train, test = seg["train"], seg["test"]
    if not train or not test:
        return {"status": "empty_segment", "rows": len(rows)}
    if len({r["move_hit"] for r in train}) < 2 or len({r["move_hit"] for r in test}) < 2:
        return {"status": "single_class", "rows": len(rows)}

    model = _fit_dual(train)
    if model.get("model_type") == "constant_dual_v1":
        return {"status": "degenerate_model", "rows": len(rows)}

    y = [int(r["move_hit"]) for r in test]
    ts = [int(r["ts"]) for r in test]
    scores = _predict_head(model, test, target="move")
    model_res = _precision_at_budget(scores, y, ts, budget, z, horizon)
    rule_res = _best_rule(test, y, ts, budget=budget, z=z, horizon=horizon)
    out: dict[str, Any] = {
        "status": "ok",
        "window": window,
        "horizon": horizon,
        "rows": len(rows),
        "train_rows": len(train),
        "test_rows": len(test),
        "base_rate": sum(y) / len(y),
        "model": model_res,
        "best_rule": rule_res,
        "verdict": _verdict(model_res, rule_res),
    }
    if importance:
        # "Which factors" — the second half of what the owner asked for. Drop
        # in PR-AUC on the test segment when each feature is shuffled.
        from amber.models.importance import permutation_importance

        imp = permutation_importance(model, test, target="move", label_key="move_hit", n_repeats=2)
        if imp.get("status") == "ok":
            out["top_factors"] = [
                {"feature": s["feature"], "importance_pct": s["importance_pct"]} for s in imp["scores"][:8]
            ]
    return out


def run_ignition_check(
    features_root: Path,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    calm_pct: float = DEFAULT_CALM_PCT,
    barrier: float = DEFAULT_BARRIER,
    budget: float = DEFAULT_BUDGET,
    max_candles_per_symbol: int = 4320,
    min_warmup_bars: int = 60,
    train_frac: float = 0.6,
    calib_frac: float = 0.15,
) -> dict[str, Any]:
    series = load_series(
        features_root,
        max_candles_per_symbol=max_candles_per_symbol,
        min_warmup_bars=min_warmup_bars,
        vol_context=max(windows),
    )
    if not series:
        return {"status": "no_features", "arms": []}

    n_arms = max(1, len(windows) * len(horizons))
    # Each arm compares the model and every feature in both directions.
    z = family_z(n_arms * (1 + 2 * len(MODEL_FEATURES)))
    arms: list[dict[str, Any]] = []
    for window in windows:
        for horizon in horizons:
            rows, considered = build_ignition_rows(
                series, window=window, horizon=horizon, calm_pct=calm_pct, barrier=barrier
            )
            res = evaluate_ignition_arm(
                rows, window=window, horizon=horizon, budget=budget,
                train_frac=train_frac, calib_frac=calib_frac, z=z,
            )
            res.update({"window": window, "horizon": horizon,
                        "calm_share": (len(rows) / considered) if considered else None})
            logger.info("ignition arm window=%s horizon=%s -> %s", window, horizon, res.get("verdict", res.get("status")))
            arms.append(res)

    scored = [a for a in arms if a.get("status") == "ok"]
    best = max(scored, key=lambda a: _VERDICT_RANK.get(a["verdict"].split(":")[0], 0), default=None)
    return {
        "status": "ok",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "symbols": len(series),
        "calm_pct": calm_pct,
        "barrier": barrier,
        "budget": budget,
        "z": z,
        "arms": arms,
        "verdict": best["verdict"] if best else "no_scored_arm",
    }


_VERDICT_RU = {
    "precursor_found": "НАЙДЕН ПРЕДВЕСТНИК: модель предсказывает ход из тишины лучше любого одного признака",
    "signal_matched_by": "предвестник есть, но один признак ловит его так же хорошо, как модель",
    "single_feature_signal": "слабый предвестник есть только в одном признаке",
    "no_precursor": "предвестника нет: на этих данных спокойный рынок не предупреждает о ходе",
    "underpowered": "мало эпизодов для вывода",
}


def format_report(report: dict[str, Any]) -> str:
    if report.get("status") != "ok":
        return f"ignition check: {report.get('status')}"

    def f(v: Any, spec: str = "{:.3f}") -> str:
        return "-" if v is None else spec.format(v)

    lines = [
        f"Зарождение движения: тишина (диапазон < {report['calm_pct'] * 100:.1f}% за окно) "
        f"→ ход ≥{report['barrier'] * 100:.1f}% в любую сторону, вход со следующей свечи",
        f"{report['symbols']} символов · бюджет {report['budget'] * 100:.1f}% тихих строк теста · "
        f"z={report['z']:.2f} (поправка на все сравнения)",
        "",
        f"{'окно':>4} {'гориз':>5} {'тихих':>6} {'база':>6} │ {'модель':>6} {'lift':>5} {'lift_lo':>7} {'эпиз':>4} │ "
        f"{'лучший признак':<22} {'lift':>5} {'lift_lo':>7} │ вывод",
    ]
    for a in report["arms"]:
        if a.get("status") != "ok":
            lines.append(f"{a['window']:>4} {a['horizon']:>5}  {a.get('status')}")
            continue
        m, r = a["model"], a["best_rule"]
        lines.append(
            f"{a['window']:>4} {a['horizon']:>5} {f(a['calm_share'], '{:.0%}'):>6} {f(a['base_rate']):>6} │ "
            f"{f(m.get('precision')):>6} {f(m.get('lift'), '{:.2f}'):>5} {f(m.get('lift_ci_low_clustered'), '{:.2f}'):>7} "
            f"{m.get('episodes', 0):>4} │ {str(r.get('rule')):<22} {f(r.get('lift'), '{:.2f}'):>5} "
            f"{f(r.get('lift_ci_low_clustered'), '{:.2f}'):>7} │ {a['verdict']}"
        )
    key = report["verdict"].split(":")[0]
    lines += ["", f"ИТОГ: {report['verdict']} — {_VERDICT_RU.get(key, '')}"]
    best = max((a for a in report["arms"] if a.get("top_factors")),
               key=lambda a: a["model"].get("lift_ci_low_clustered") or 0.0, default=None)
    if best:
        lines += ["", f"Факторы модели (окно {best['window']}, горизонт {best['horizon']}), падение PR-AUC при перемешивании:"]
        lines += [f"  {t['feature']:<20} {t['importance_pct']:7.2f}%" for t in best["top_factors"]]
    return "\n".join(lines)


def save_report(logs_root: Path, report: dict[str, Any]) -> Path:
    logs_root.mkdir(parents=True, exist_ok=True)
    path = logs_root / REPORT_FILE
    path.write_text(json.dumps(report, ensure_ascii=False, default=str), encoding="utf-8")
    return path
