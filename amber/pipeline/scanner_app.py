from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from amber.alerts.router import AlertRateLimiter, route_alert
from amber.common.config import ConfigLoader
from amber.common.logging import setup_logging
from amber.common.types import SignalV1
from amber.signals.filters import SignalGate, passes_thresholds
from amber.signals.scorer import score_signal
from amber.signals.universe import select_universe

logger = logging.getLogger(__name__)


def _read_latest_feature_rows(features_root: Path, allowed_symbols: set[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file in sorted((features_root / "features").glob("*/part-000.jsonl")):
        symbol = file.parent.name
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




def _build_alert_limiter(thresholds_cfg: dict[str, Any]) -> AlertRateLimiter:
    cooldown = int(thresholds_cfg.get("cooldown_sec", 0))
    return AlertRateLimiter(cooldown_sec=max(0, cooldown))
def main() -> None:
    config = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    thresholds = ConfigLoader(Path.cwd()).load_yaml("config/thresholds.yaml")
    setup_logging(config.get("run", {}).get("log_level", "INFO"))

    features_root = Path(config["storage"]["features_dir"])
    models_root = Path(config["storage"]["models_dir"])
    logs_root = Path(config["storage"]["logs_dir"])
    alert_channels = config.get("alerts", {}).get("channels", ["console"])

    top_k = int(config.get("signal", {}).get("top_k_universe", 20))
    universe = set(select_universe(features_root, top_k=top_k, min_obs=1))

    gate = SignalGate(
        cooldown_sec=int(thresholds["thresholds"].get("cooldown_sec", 90)),
        concurrent_limit=int(thresholds["thresholds"].get("concurrent_limit", 5)),
    )

    feature_rows = _read_latest_feature_rows(features_root, allowed_symbols=universe)
    alert_limiter = _build_alert_limiter(thresholds["thresholds"])
    emitted = 0
    for row in feature_rows:
        signal = score_signal(row, models_root=models_root, config_version=config["signal"]["schema_version"])
        if passes_thresholds(
            signal,
            up_min=float(thresholds["thresholds"]["pump_prob_calibrated_min"]),
            down_min=float(thresholds["thresholds"]["dump_prob_calibrated_min"]),
            directional_min=float(thresholds["thresholds"].get("directional_score_min", 0.2)),
            spread_max_bps=float(thresholds["thresholds"].get("spread_bps_max", 30.0)),
        ) and gate.allow(signal):
            _append_signal(logs_root, signal)
            route_alert(signal, channels=alert_channels, limiter=alert_limiter)
            emitted += 1

    logger.info("scanner finished universe=%s symbols=%s emitted=%s", len(universe), len(feature_rows), emitted)


if __name__ == "__main__":
    main()
