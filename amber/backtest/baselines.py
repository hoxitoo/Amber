"""Is the model a model, or a rebadged indicator? (roadmap D11)

`range_atr_14` carries 85% of permutation importance and the live drivers read
`range_atr_14=+4.2` — the model fires when volatility is *already* four sigma
high. "Price is moving violently right now" predicting "price will move 1% in
the next 15 minutes" is close to a tautology: true, and almost free of
forecasting content. Precision 0.98 is then a statement about how easy the
question became, not about the model.

The test is a trivial baseline: rank rows by one raw feature and alert on the
top N, the same N the model gets. Three things are reported, and the order
matters:

1. **Overlap** — of the model's alerts, how many the single-feature rule also
   picks. This is the robust one: it involves no outcomes at all, so it does not
   care that the test segment holds only a handful of independent episodes. At
   high overlap the model *is* the rule, whatever the precisions say.
2. **Rank correlation** between the two scores, over the union of their alert
   sets. Informational only — it is a weak discriminator and no verdict rests
   on it. Measured on a fixture where one feature drove the label: membership
   overlap 77% while the correlation was 0.05, because both agree on *which*
   rows are extreme and disagree on the ordering *within* the extreme tail.
   Those are different questions, and only the first says "same tool".
3. **Precision and lift**, for completeness, with the same episode-clustered
   bound used everywhere else.

Each baseline is evaluated in both directions and credited with its better one,
so the comparison does not flatter the model by handicapping the rule.

Read-only.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Sequence

from amber.backtest.label_sweep import _episodes, _wilson_low, family_z
from amber.backtest.operating_point import _label, LABEL_KEYS

logger = logging.getLogger(__name__)

# The features worth testing as stand-alone rules: the ones permutation
# importance says the model leans on.
DEFAULT_BASELINES = ("range_atr_14", "bb_width_20", "vol_z_20", "vol_ratio_20", "spread_bps")

# Above this share of shared alerts, the model and the rule are the same tool.
TAUTOLOGY_OVERLAP = 0.80
MIN_EPISODES = 10


def _spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Rank correlation, computed without scipy."""
    n = len(a)
    if n < 3:
        return None

    def ranks(xs: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: xs[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            shared = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((x - mb) ** 2 for x in rb)
    if va <= 0 or vb <= 0:
        return None
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / (va**0.5 * vb**0.5)


def _score_set(scores: Sequence[float], take: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:take]


def _measure(
    chosen: Sequence[int],
    labels: Sequence[int],
    ts: Sequence[int],
    base: float,
    horizon: int,
    z: float,
) -> dict[str, Any]:
    take = len(chosen)
    hits = sum(labels[i] for i in chosen)
    precision = hits / take if take else 0.0
    eps = _episodes([ts[i] for i in chosen], horizon)
    low = _wilson_low(int(round(precision * eps)), eps, z) if eps else 0.0
    return {
        "alerts": take,
        "precision": precision,
        "lift": (precision / base) if base > 0 else None,
        "episodes": eps,
        "lift_ci_low_clustered": (low / base) if base > 0 else None,
        "underpowered": eps < MIN_EPISODES,
    }


def compare_to_baselines(
    datasets_root: Path,
    models_root: Path,
    *,
    target: str = "move",
    rate_per_day: float = 400.0,
    features: Sequence[str] = DEFAULT_BASELINES,
    seed: int = 11,
) -> dict[str, Any]:
    """The model against one-feature rules, on identical rows and alert counts."""
    from amber.models.dataset_io import load_latest_dataset_rows, order_with_pseudo_time, split_rows
    from amber.models.importance import _clipped_matrix, _predict_matrix
    from amber.models.infer import load_latest_model
    from amber.signals.scorer import _load_latest_calibration, calibrated_prob_for_target

    if target not in LABEL_KEYS:
        raise ValueError(f"target must be one of {sorted(LABEL_KEYS)}, got {target!r}")

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

    ordered, pts, _mode = order_with_pseudo_time(all_rows)
    splits = model.get("splits")
    rows = split_rows(ordered, pts, splits)["test"] if isinstance(splits, dict) else ordered
    if not rows:
        return {"status": "empty_test_segment"}

    labels = [_label(r, target) for r in rows]
    base = sum(labels) / len(labels)
    if base <= 0:
        return {"status": "no_positive_labels", "rows": len(rows)}

    calibration = _load_latest_calibration(models_root)
    matrix, _names = _clipped_matrix(model, rows)
    model_scores = [
        calibrated_prob_for_target(p, calibration=calibration, target=target)
        for p in _predict_matrix(model, target, matrix)
    ]

    ts = [int(r.get("ts", 0) or 0) for r in rows]
    span_days = max(1e-9, (max(ts) - min(ts)) / 86_400_000.0)
    horizon = int(rows[0].get("horizon_steps", 15) or 15)
    take = max(1, min(len(rows), int(round(rate_per_day * span_days))))
    z = family_z(len(features) + 1)

    model_set = _score_set(model_scores, take)
    out: dict[str, Any] = {
        "status": "ok",
        "target": target,
        "dataset_run": dataset_run,
        "rows": len(rows),
        "base_rate": base,
        "span_days": span_days,
        "horizon": horizon,
        "alerts_per_day": rate_per_day,
        "family_z": z,
        "model": {"name": "model", **_measure(model_set, labels, ts, base, horizon, z)},
    }

    rng = random.Random(seed)
    model_ids = set(model_set)
    baselines: list[dict[str, Any]] = []

    for name in features:
        values = [float(r.get(name, 0.0) or 0.0) for r in rows]
        if len(set(values)) < 2:
            baselines.append({"name": name, "status": "constant"})
            continue
        # Credit the rule with its better direction: a feature where low values
        # mean "about to move" would otherwise be scored as useless by an
        # arbitrary sign convention, understating the baseline.
        best = None
        for sign, label in ((1.0, "high"), (-1.0, "low")):
            picks = _score_set([sign * v for v in values], take)
            res = _measure(picks, labels, ts, base, horizon, z)
            if best is None or (res["precision"] or 0) > (best["precision"] or 0):
                best = {**res, "direction": label, "_picks": picks}
        assert best is not None
        picks = best.pop("_picks")
        shared = len(model_ids & set(picks))
        # Correlation over the union of the two alert sets, not over every row.
        # Across all rows the mass of quiet, uninformative candles dominates the
        # ranks and drags the coefficient toward zero even when the two agree
        # completely on the rows either of them would fire on — measured at 0.11
        # for a model that was picking 77% of the same alerts.
        union = sorted(model_ids | set(picks))
        sign = 1.0 if best["direction"] == "high" else -1.0
        out_corr = (
            _spearman([model_scores[i] for i in union], [sign * values[i] for i in union])
            if len(union) >= 3
            else None
        )
        baselines.append({
            "name": name,
            "status": "ok",
            **best,
            "overlap_with_model": shared / take if take else 0.0,
            "shared_alerts": shared,
            "spearman_with_model": out_corr,
        })

    random_scores = [rng.random() for _ in rows]
    out["random"] = {"name": "random", **_measure(_score_set(random_scores, take), labels, ts, base, horizon, z)}
    out["baselines"] = baselines
    out["verdict"] = _verdict(out)
    return out


def _verdict(report: dict[str, Any]) -> str:
    """What the comparison supports, not what it happens to show.

    The first version returned `model_adds_signal` whenever the model's point
    precision exceeded every rule's. On the live run that declared victory on a
    difference of ONE alert in 103 — model 1.000 against range_atr_14 0.990 —
    while the model's own clustered lower bound (5.67) sat well below the rule's
    point lift (9.63), and `spread_bps` actually had the *better* bound. A margin
    no statistics supports is not an advantage, and the tool exists precisely to
    stop the project acting on that kind of number.
    """
    ok = [b for b in report["baselines"] if b.get("status") == "ok"]
    if not ok:
        return "no_baselines"

    twin = max(ok, key=lambda b: b.get("overlap_with_model") or 0.0)
    if (twin.get("overlap_with_model") or 0.0) >= TAUTOLOGY_OVERLAP:
        return f"tautology:{twin['name']}"

    model = report["model"]
    strongest = max(ok, key=lambda b: b.get("precision") or 0.0)
    model_low = model.get("lift_ci_low_clustered") or 0.0
    best_lift = strongest.get("lift") or 0.0

    # The model only "adds" something if what the data supports for it clears
    # what the rule actually achieved.
    if model_low > best_lift:
        return "model_adds_signal"
    if (strongest.get("precision") or 0.0) >= (model.get("precision") or 0.0):
        return f"matched_by:{strongest['name']}"
    return f"indistinguishable_from:{strongest['name']}"


def format_report(report: dict[str, Any]) -> str:
    if report.get("status") != "ok":
        return f"baseline comparison unavailable: {report.get('status')} {report.get('detail', '')}".strip()

    def _f(v: Any, spec: str = "{:.3f}") -> str:
        return spec.format(v) if isinstance(v, (int, float)) else "—"

    lines = [
        f"target {report['target']} · base rate {report['base_rate']:.4f} · "
        f"{report['rows']:,} test rows over {report['span_days']:.2f} days · "
        f"{report['alerts_per_day']:.0f} alerts/day = {report['model']['alerts']} alerts each",
        "",
        f"{'правило':<18} {'precision':>10} {'lift':>7} {'lift_lo':>8} {'эпизодов':>9} "
        f"{'совпало с моделью':>18} {'rank corr':>10}",
        "-" * 88,
    ]
    m = report["model"]
    lines.append(
        f"{'МОДЕЛЬ':<18} {_f(m['precision']):>10} {_f(m['lift'], '{:.2f}'):>7} "
        f"{_f(m['lift_ci_low_clustered'], '{:.2f}'):>8} {_f(m['episodes'], '{:.0f}'):>9} "
        f"{'—':>18} {'—':>10}"
    )
    for b in report["baselines"]:
        if b.get("status") != "ok":
            lines.append(f"{b['name']:<18} {b.get('status')}")
            continue
        lines.append(
            f"{b['name']:<18} {_f(b['precision']):>10} {_f(b['lift'], '{:.2f}'):>7} "
            f"{_f(b['lift_ci_low_clustered'], '{:.2f}'):>8} {_f(b['episodes'], '{:.0f}'):>9} "
            f"{_f(b['overlap_with_model'], '{:.0%}'):>18} {_f(b['spearman_with_model'], '{:+.2f}'):>10}"
        )
    r = report["random"]
    lines.append(
        f"{'случайно':<18} {_f(r['precision']):>10} {_f(r['lift'], '{:.2f}'):>7} "
        f"{_f(r['lift_ci_low_clustered'], '{:.2f}'):>8} {_f(r['episodes'], '{:.0f}'):>9} "
        f"{'—':>18} {'—':>10}"
    )

    verdict = report["verdict"]
    lines += ["", f"verdict: {verdict}"]
    if verdict.startswith("tautology:"):
        feature = verdict.split(":", 1)[1]
        lines.append(
            f"Модель выбирает те же строки, что правило «{feature} выше порога». "
            "Это индикатор в обёртке ML: обучение, калибровка и пороги ничего не добавляют "
            "поверх одной фичи. Улучшать надо не модель, а входные данные — нужен сигнал, "
            "опережающий сам всплеск (roadmap D2/D7)."
        )
    elif verdict.startswith("matched_by:"):
        feature = verdict.split(":", 1)[1]
        lines.append(
            f"Множества алертов различаются, но точность у «{feature}» не ниже. "
            "Модель не проигрывает, но и не окупает свою сложность на этих данных."
        )
    elif verdict.startswith("indistinguishable_from:"):
        feature = verdict.split(":", 1)[1]
        lines.append(
            f"Модель показала точность чуть выше, чем «{feature}», но её собственная "
            "нижняя граница ниже того, чего правило фактически достигло — разница "
            "статистикой не подтверждена. Считать это превосходством нельзя."
        )
    elif verdict == "model_adds_signal":
        lines.append(
            "Модель обходит каждое однофичевое правило с запасом, переживающим поправку "
            "на слипание, и выбирает другие строки. Это то, что можно развивать."
        )

    majority = [b for b in report["baselines"]
                if b.get("status") == "ok" and (b.get("overlap_with_model") or 0) >= 0.5]
    if majority and not verdict.startswith("tautology:"):
        names = ", ".join(f"{b['name']} ({b['overlap_with_model']:.0%})" for b in majority)
        lines.append(
            f"Отдельно: большинство алертов модели выбирают и простые правила — {names}. "
            "Ниже порога тавтологии, но это не независимый инструмент."
        )
    lines.append(
        "Главная колонка — совпадение с моделью: она не зависит от исходов, поэтому "
        "не страдает от малого числа эпизодов, в отличие от precision."
    )
    return "\n".join(lines)


def save_report(logs_root: Path, report: dict[str, Any]) -> Path:
    logs_root.mkdir(parents=True, exist_ok=True)
    out = logs_root / "baseline_comparison.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
