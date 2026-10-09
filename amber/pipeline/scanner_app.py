from __future__ import annotations

import logging
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

from amber.alerts.router import AlertRateLimiter, route_alert
from amber.common.audit_log import log_universe
from amber.common.config import ConfigLoader
from amber.common.jsonl import read_last
from amber.common.locks import AlreadyRunning, SingleInstanceLock
from amber.common.logging import setup_logging
from amber.common.types import SignalV1
from amber.models.infer import load_latest_model
from amber.signals.filters import SignalGate, base_rate_for, effective_prob_min, passes_thresholds
from amber.signals.scorer import _load_latest_calibration, score_signal
from amber.signals.shadow import GATE_STATE_KEY, append_shadow_signal, load_shadow_rule, shadow_candidates
from amber.signals.universe import select_universe
from amber.storage.state_store import StateStore

logger = logging.getLogger(__name__)


def _read_latest_feature_rows(features_root: Path, allowed_symbols: set[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_symbol: dict[str, Path] = {}
    for file in sorted((features_root / "features").glob("*/part-*.jsonl")):
        by_symbol[file.parent.name] = file  # keep the latest part per symbol

    for symbol, file in sorted(by_symbol.items()):
        if allowed_symbols is not None and symbol not in allowed_symbols:
            continue
        # Stream the tail: these files hold tens of thousands of rows and this
        # runs for every symbol on every scan, so reading them whole made the
        # scan cost grow with accumulated history.
        parsed = read_last(file)
        if parsed is None:
            logger.warning("skip feature file with invalid JSONL tail: %s", file)
            continue
        rows.append(parsed)
    return rows


def _gating_prob(signal: SignalV1, on_move: bool) -> float:
    """The probability the gate actually decided on, used to rank candidates.

    Ranking by something other than the gating quantity would order the alert
    list by one criterion while admitting it by another.
    """
    if on_move:
        move = signal.prob_move_calibrated
        if move is not None:
            return float(move)
        return min(1.0, signal.prob_up_calibrated + signal.prob_down_calibrated)
    return max(signal.prob_up_calibrated, signal.prob_down_calibrated)


def _append_signal(logs_root: Path, signal: SignalV1) -> None:
    logs_root.mkdir(parents=True, exist_ok=True)
    out = logs_root / "signals.jsonl"
    with out.open("a", encoding="utf-8") as fh:
        fh.write(signal.model_dump_json() + "\n")


def scan_once(
    config: dict[str, Any],
    thresholds: dict[str, Any],
    gate: SignalGate,
    alert_limiter: AlertRateLimiter,
    shadow_gate: SignalGate | None = None,
    ignition_state: StateStore | None = None,
) -> int:
    features_root = Path(config["storage"]["features_dir"])
    models_root = Path(config["storage"]["models_dir"])
    logs_root = Path(config["storage"]["logs_dir"])
    alert_channels = config.get("alerts", {}).get("channels", ["console"])

    signal_cfg = config.get("signal", {})
    top_k = int(signal_cfg.get("top_k_universe", 20))
    min_dollar_volume = float(signal_cfg.get("min_dollar_volume", 0.0))
    min_warmup = int(config.get("labeling", {}).get("min_warmup_bars", 60))
    universe = set(
        select_universe(features_root, top_k=top_k, min_obs=min_warmup, min_dollar_volume=min_dollar_volume)
    )
    log_universe(logs_root, "scanner", sorted(universe))

    try:
        model = load_latest_model(models_root)
    except (ValueError, FileNotFoundError, OSError):
        logger.warning("no trained model yet under %s; waiting (run training first)", models_root)
        return 0
    calibration = _load_latest_calibration(models_root)

    # Base-rate-relative operating thresholds (audit B3): a well-calibrated head
    # for a rare event rarely exceeds a fixed 0.65, so the absolute cut would
    # never fire. Derive the cut from the head's own base rate instead.
    up_min = effective_prob_min(thresholds, base_rate_for(model, "pump"), absolute_key="pump_prob_calibrated_min")
    down_min = effective_prob_min(thresholds, base_rate_for(model, "dump"), absolute_key="dump_prob_calibrated_min")

    # Gate on movement when the model has a `move` head (roadmap D10). Older
    # models have no such head and keep the directional gate, so a rollback or a
    # stale artifact degrades to the previous behaviour rather than firing on
    # everything.
    heads = model.get("heads", {})
    move_min = None
    if isinstance(heads, dict) and "move" in heads:
        move_min = effective_prob_min(
            thresholds, base_rate_for(model, "move"), absolute_key="move_prob_calibrated_min"
        )

    feature_rows = _read_latest_feature_rows(features_root, allowed_symbols=universe)

    candidates: list[tuple[float, SignalV1]] = []
    for row in feature_rows:
        if bool(row.get("is_synthetic", False)):
            continue  # do not signal on gap-filled candles
        if int(row.get("obs", 0) or 0) < min_warmup:
            continue  # not enough history: long-lookback features are still degenerate
        signal = score_signal(
            row,
            models_root=models_root,
            config_version=config["signal"]["schema_version"],
            model=model,
            calibration=calibration,
        )
        if passes_thresholds(
            signal,
            up_min=up_min,
            down_min=down_min,
            directional_min=float(thresholds.get("directional_score_min", 0.2)),
            spread_max_bps=float(thresholds.get("spread_bps_max", 30.0)),
            move_min=move_min,
        ):
            candidates.append((_gating_prob(signal, move_min is not None), signal))

    # Strongest first. `SignalGate` hands out its `concurrent_limit` slots on a
    # first-come basis, and rows arrive sorted by symbol, so the slots were going
    # to whichever symbols come first ALPHABETICALLY among those over the
    # threshold — ASTERUSDT ahead of ZECUSDT regardless of which the model
    # actually preferred. With a loose threshold most of the universe qualifies,
    # so the alert list was largely alphabetical rather than ranked.
    candidates.sort(key=lambda pair: pair[0], reverse=True)

    emitted = 0
    for _prob, signal in candidates:
        if gate.allow(signal):
            # When the alert actually left, as opposed to the bar it describes:
            # the pipeline and the scan each run once a minute, so an alert can
            # reach the owner two to three minutes after its bar opened. The
            # ledger enters after this moment, not after the bar.
            signal.market_context["emitted_ms"] = int(time.time() * 1000)
            _append_signal(logs_root, signal)
            route_alert(signal, channels=alert_channels, limiter=alert_limiter)
            emitted += 1

    shadow_emitted = _scan_shadow(
        logs_root, feature_rows, model, shadow_gate,
        min_warmup=min_warmup, spread_max_bps=float(thresholds.get("spread_bps_max", 30.0)),
    )
    _scan_ignition(config, features_root, models_root, logs_root, ignition_state)

    logger.info(
        "scan finished universe=%s symbols=%s emitted=%s shadow=%s gate=%s",
        len(universe), len(feature_rows), emitted, shadow_emitted,
        f"move>={move_min:.4f}" if move_min is not None else f"dir up>={up_min:.4f}/down>={down_min:.4f}",
    )
    return emitted


def _scan_shadow(
    logs_root: Path,
    feature_rows: list[dict[str, Any]],
    model: dict[str, Any],
    gate: SignalGate | None,
    *,
    min_warmup: int,
    spread_max_bps: float,
) -> int:
    """Run the range_atr_14 rule next to the model for the forward ledger.

    Never allowed to break the real scan: it is measurement, not product.
    """
    if gate is None:
        return 0
    try:
        rule = load_shadow_rule(logs_root)
        if rule is None:
            return 0
        labeling = model.get("labeling", {}) if isinstance(model.get("labeling"), dict) else {}
        horizon = int(labeling.get("horizon_steps", 15))
        target = float(labeling.get("avg_up_pct", 0.01))
        emitted = 0
        for row in shadow_candidates(feature_rows, rule, min_warmup=min_warmup, spread_max_bps=spread_max_bps):
            probe = SimpleNamespace(symbol=row["symbol"], event_ts=int(row["ts"]) / 1000.0)
            if gate.allow(probe):
                append_shadow_signal(logs_root, row, rule, horizon_min=horizon, target_pct=target,
                                     emitted_ms=int(time.time() * 1000))
                emitted += 1
        return emitted
    except Exception as exc:
        logger.warning("shadow scan failed: %s", exc)
        return 0


_IGNITION_CACHE: dict[str, Any] = {"path": None, "scorer": None, "waiting_logged": False}


def _scan_ignition(
    config: dict[str, Any],
    features_root: Path,
    models_root: Path,
    logs_root: Path,
    state: StateStore | None,
) -> int:
    """Ignition warnings, a shadow channel beside the live scan.

    Never allowed to break the real scan. The scorer is cached until a newer
    artifact appears, so the booster is not rebuilt every minute.
    """
    ign = config.get("ignition", {}) if isinstance(config.get("ignition"), dict) else {}
    if state is None or not ign.get("enabled", False):
        return 0
    try:
        from amber.signals.ignition_live import IgnitionScorer, alert_text, latest_artifact_path, scan_ignition

        path = latest_artifact_path(models_root)
        if path is None:
            # Said once, so an empty log reads as "waiting", not "broken".
            if not _IGNITION_CACHE["waiting_logged"]:
                logger.info("ignition: no model yet, waiting for the daily training in amber-pipeline")
                _IGNITION_CACHE["waiting_logged"] = True
            return 0
        if _IGNITION_CACHE["path"] != path:
            _IGNITION_CACHE.update({"path": path, "scorer": IgnitionScorer.load_latest(models_root)})
            logger.info("ignition: model loaded %s", path.name)
        scorer = _IGNITION_CACHE["scorer"]
        if scorer is None:
            logger.warning("ignition: model %s could not be loaded", path.name)
            return 0
        records = scan_ignition(features_root, models_root, logs_root, state, scorer=scorer)
        notify = [r for r in records if r.get("notify")]
        if notify and ign.get("telegram", False):
            from amber.alerts.telegram import send_telegram_text

            for rec in notify:
                send_telegram_text(alert_text(rec))
        # Every scan, zeros included: silence would be indistinguishable from
        # a scanner that never reached this code.
        logger.info("ignition: %s calm coins scored, %s alerts, %s shown", len(records),
                    sum(r["alert"] for r in records), len(notify))
        return len(notify)
    except Exception as exc:
        logger.warning("ignition scan failed: %s", exc)
        return 0


def main(loop: bool = False) -> None:
    config = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    thresholds_cfg = ConfigLoader(Path.cwd()).load_yaml("config/thresholds.yaml")["thresholds"]
    setup_logging(config.get("run", {}).get("log_level", "INFO"))

    state = StateStore(Path(config["storage"]["state_dir"]))
    horizon_min = int(config.get("labeling", {}).get("horizon_steps", 5))
    gate = SignalGate(
        cooldown_sec=int(thresholds_cfg.get("cooldown_sec", 90)),
        concurrent_limit=int(thresholds_cfg.get("concurrent_limit", 5)),
        slot_ttl_sec=max(60, horizon_min * 60),
        store=state,
    )
    # Same limits, its own state: the rule must not take the model's slots.
    shadow_gate = SignalGate(
        cooldown_sec=int(thresholds_cfg.get("cooldown_sec", 90)),
        concurrent_limit=int(thresholds_cfg.get("concurrent_limit", 5)),
        slot_ttl_sec=max(60, horizon_min * 60),
        store=state,
        state_key=GATE_STATE_KEY,
    )
    alert_limiter = AlertRateLimiter(cooldown_sec=max(0, int(thresholds_cfg.get("cooldown_sec", 0))), store=state)

    interval = max(5, int(config.get("scanner", {}).get("interval_sec", 60)))
    try:
        with SingleInstanceLock(Path(config["storage"]["state_dir"]) / "locks", "scanner"):
            while True:
                scan_once(config, thresholds_cfg, gate, alert_limiter, shadow_gate, ignition_state=state)
                if not loop:
                    break
                time.sleep(interval)
    except AlreadyRunning:
        logger.warning("scanner already running; skipping")


if __name__ == "__main__":
    import sys

    main(loop="--loop" in sys.argv)
