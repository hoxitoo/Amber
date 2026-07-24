from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any

from amber.alerts.router import AlertRateLimiter, route_alert
from amber.common.audit_log import log_universe
from amber.common.config import ConfigLoader
from amber.common.locks import AlreadyRunning, SingleInstanceLock
from amber.common.logging import setup_logging
from amber.common.types import SignalV1
from amber.models.infer import load_latest_model
from amber.signals.filters import SignalGate, base_rate_for, effective_prob_min, passes_thresholds
from amber.signals.scorer import _load_latest_calibration, score_signal
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
        lines = file.read_text(encoding="utf-8").splitlines()
        if not lines:
            continue

        parsed: dict[str, Any] | None = None
        for raw in reversed(lines):
            try:
                parsed = json.loads(raw)
                break
            except json.JSONDecodeError:
                continue
        if parsed is None:
            logger.warning("skip feature file with invalid JSONL tail: %s", file)
            continue
        rows.append(parsed)
    return rows


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

    feature_rows = _read_latest_feature_rows(features_root, allowed_symbols=universe)
    emitted = 0
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
        ) and gate.allow(signal):
            _append_signal(logs_root, signal)
            route_alert(signal, channels=alert_channels, limiter=alert_limiter)
            emitted += 1

    logger.info("scan finished universe=%s symbols=%s emitted=%s", len(universe), len(feature_rows), emitted)
    return emitted


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
    alert_limiter = AlertRateLimiter(cooldown_sec=max(0, int(thresholds_cfg.get("cooldown_sec", 0))), store=state)

    interval = max(5, int(config.get("scanner", {}).get("interval_sec", 60)))
    try:
        with SingleInstanceLock(Path(config["storage"]["state_dir"]) / "locks", "scanner"):
            while True:
                scan_once(config, thresholds_cfg, gate, alert_limiter)
                if not loop:
                    break
                time.sleep(interval)
    except AlreadyRunning:
        logger.warning("scanner already running; skipping")


if __name__ == "__main__":
    import sys

    main(loop="--loop" in sys.argv)
