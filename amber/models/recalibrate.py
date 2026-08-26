"""Calibration health and rolling recalibration (audit M5).

A retrain fits calibration on the segment 70-85% of the way through the training
window — with a 48h window that is data roughly 7-14 hours old. That is fine
while the market is stationary and wrong when it is not: the observed event rate
here moved from ~10% to ~22% (pump) and ~7% to ~28% (dump) inside two weeks, and
a calibration fitted before the shift maps scores to the old frequencies. The
symptom is a persistent prediction bias, which then feeds the base-rate-relative
gate and the reported confidence.

Refitting calibration is cheap — two coefficients on a few thousand rows — so it
can run far more often than a full retrain, which is the whole point of a
separate cadence.

Honesty note: recalibrating on the freshest rows means calibration-dependent
metrics (Brier, precision at threshold) computed on those same rows are no
longer out-of-sample, so the result is flagged. Ranking metrics (AUC, PR-AUC) are
unaffected, since a monotone calibration cannot reorder scores.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RECAL_FILE = "calibration_health.json"
DEFAULT_TAIL_ROWS = 40_000
# Above this expected-calibration-error the mapping is stale enough to refit.
ECE_REFIT = 0.05
BIAS_REFIT = 0.05


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, float(p)))
    return math.log(p / (1.0 - p))


def read_dataset_tail(datasets_root: Path, max_rows: int = DEFAULT_TAIL_ROWS) -> list[dict[str, Any]]:
    """Newest rows only, streamed with a bounded buffer.

    Recalibration runs often, so it must not pull the whole several-hundred-MB
    dataset into memory the way a retrain does.
    """
    from amber.models.dataset_io import latest_dataset_dir

    try:
        path = latest_dataset_dir(datasets_root) / "dataset.jsonl"
    except ValueError:
        return []  # nothing built yet — a scheduled check must not raise
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        tail = deque(fh, maxlen=max_rows)
    rows: list[dict[str, Any]] = []
    for line in tail:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def calibration_error(probs: list[float], labels: list[int], bins: int = 10) -> dict[str, float]:
    """Expected calibration error plus the raw predicted-vs-observed gap.

    ECE is the average distance between what the model promised and what
    happened, weighted by how many predictions fell in each bucket — i.e. how
    much "70%" can be trusted to mean 70%.
    """
    if not probs:
        return {"ece": 0.0, "bias": 0.0, "mean_pred": 0.0, "observed": 0.0, "n": 0}
    n = len(probs)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (p >= lo and (p < hi or (b == bins - 1 and p <= hi)))]
        if not idx:
            continue
        mean_pred = sum(probs[i] for i in idx) / len(idx)
        observed = sum(labels[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(mean_pred - observed)
    mean_pred = sum(probs) / n
    observed = sum(labels) / n
    return {
        "ece": ece,
        "bias": mean_pred - observed,
        "mean_pred": mean_pred,
        "observed": observed,
        "n": n,
    }


def _fit_platt(raw: list[float], y: list[int]) -> dict[str, Any] | None:
    if len(set(y)) < 2 or len(raw) < 50:
        return None
    try:
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(max_iter=1000)
        lr.fit([[_logit(s)] for s in raw], y)
        return {"method": "platt", "a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])}
    except Exception as exc:
        logger.warning("recalibration fit failed: %s", exc)
        return None


def check_and_recalibrate(
    models_root: Path,
    datasets_root: Path,
    *,
    tail_rows: int = DEFAULT_TAIL_ROWS,
    ece_threshold: float = ECE_REFIT,
    bias_threshold: float = BIAS_REFIT,
    apply: bool = True,
) -> dict[str, Any]:
    """Measure calibration on the freshest rows and refit it when it has drifted."""
    from amber.models.eval import _load_latest_calibration
    from amber.models.infer import infer_row_prob, load_latest_model
    from amber.signals.scorer import calibrated_prob_for_target

    try:
        model = load_latest_model(models_root)
    except Exception as exc:
        return {"status": "no_model", "reason": str(exc)}

    rows = read_dataset_tail(datasets_root, max_rows=tail_rows)
    if not rows:
        return {"status": "no_rows"}

    # One horizon only: mixing them would blend different questions.
    horizons = sorted({int(r.get("horizon_steps", 0) or 0) for r in rows})
    horizon = horizons[0] if horizons else 0
    rows = [r for r in rows if int(r.get("horizon_steps", 0) or 0) == horizon]
    if len(rows) < 200:
        return {"status": "not_enough_rows", "rows": len(rows)}

    calib = _load_latest_calibration(models_root)
    heads = calib.get("heads", {}) if isinstance(calib.get("heads"), dict) else {}
    out_heads: dict[str, Any] = {}
    report: dict[str, Any] = {}
    refit_any = False

    for target, label_key in (("pump", "up_hit"), ("dump", "down_hit")):
        raw = [infer_row_prob(model, r, target=target) for r in rows]
        y = [int(r.get(label_key, 0)) for r in rows]
        current = [calibrated_prob_for_target(p, calib, target=target) for p in raw]
        before = calibration_error(current, y)

        head_report: dict[str, Any] = {"before": before, "refit": False}
        needs = before["ece"] > ece_threshold or abs(before["bias"]) > bias_threshold
        if needs:
            fitted = _fit_platt(raw, y)
            if fitted is not None:
                after = calibration_error([calibrated_prob_for_target(p, {**fitted}, target=target) for p in raw], y)
                # Only adopt a refit that actually improves the mapping.
                if after["ece"] <= before["ece"]:
                    out_heads[target] = fitted
                    head_report.update({"refit": True, "after": after})
                    refit_any = True
                else:
                    head_report["after"] = after
                    head_report["rejected"] = "refit did not reduce ECE"
        if target not in out_heads and isinstance(heads.get(target), dict):
            out_heads[target] = heads[target]
        report[target] = head_report

    result: dict[str, Any] = {
        "status": "ok",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "horizon_steps": horizon,
        "ece_threshold": ece_threshold,
        "bias_threshold": bias_threshold,
        "refit": refit_any,
        "heads": report,
        # Metrics that depend on calibration are no longer out-of-sample on
        # these rows once a refit lands here.
        "calibration_in_sample": refit_any,
    }

    if refit_any and apply:
        payload = {
            "method": "multi_head",
            "heads": out_heads,
            "model_run_id": calib.get("model_run_id", "unknown"),
            "rows": len(rows),
            "source": "rolling_recalibration",
            "recalibrated_at": result["computed_at"],
        }
        _write_calibration(models_root, payload)
        result["written"] = True
    return result


def _write_calibration(models_root: Path, payload: dict[str, Any]) -> None:
    """Publish as a new calib_* run so the scanner picks it up by normal lookup."""
    from amber.common.manifest import new_run_id

    run_id = new_run_id(prefix="calib")
    out_dir = Path(models_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "calibration.json.tmp"
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_dir / "calibration.json")
    finally:
        tmp.unlink(missing_ok=True)
    logger.info("rolling recalibration written run_id=%s rows=%s", run_id, payload.get("rows"))


def save_health(logs_dir: Path, result: dict[str, Any]) -> None:
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / RECAL_FILE
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def load_health(logs_dir: Path) -> dict[str, Any] | None:
    path = Path(logs_dir) / RECAL_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None
