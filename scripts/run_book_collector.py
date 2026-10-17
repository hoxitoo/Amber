"""Order book + sub-minute flow collector (roadmap D2). Long-running service.

    python scripts/run_book_collector.py

Runs as systemd `amber-book-collector` (deploy/systemd/), separate from the
candle collector. Turn it off with `book.enabled: false` in config/amber.yaml.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader  # noqa: E402
from amber.common.locks import AlreadyRunning, SingleInstanceLock  # noqa: E402
from amber.common.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    cfg = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    setup_logging(cfg.get("run", {}).get("log_level", "INFO"))
    book = cfg.get("book", {}) if isinstance(cfg.get("book"), dict) else {}
    if not book.get("enabled", False):
        logger.info("book collector disabled (book.enabled is false); exiting")
        return 0
    from amber.pipeline.book_app import run

    try:
        with SingleInstanceLock(Path(cfg["storage"]["state_dir"]) / "locks", "book_collector"):
            asyncio.run(run(cfg))
    except AlreadyRunning as exc:
        logger.warning("book collector already running; not starting another (%s)", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
