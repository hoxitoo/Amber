"""Continuously turn accumulating raw WS data into fresh features.

The WS collector only *accumulates* raw payloads; this loop periodically runs
normalize -> features so the scanner and dashboard always see current candles
without any manual clicking. Start/stop it from the dashboard control panel.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
from amber.common.logging import setup_logging
from amber.pipeline import features_app, normalize_app

logger = logging.getLogger(__name__)


def main() -> None:
    cfg = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    setup_logging(cfg.get("run", {}).get("log_level", "INFO"))
    interval = max(15, int(cfg.get("pipeline", {}).get("loop_sec", 60)))
    logger.info("pipeline loop started interval=%ss", interval)
    while True:
        for stage, fn in (("normalize", normalize_app.main), ("features", features_app.main)):
            try:
                fn()
            except Exception as exc:  # keep the loop alive across transient errors
                logger.error("pipeline stage %s failed: %s", stage, exc)
        time.sleep(interval)


if __name__ == "__main__":
    main()
