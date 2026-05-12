from __future__ import annotations

import logging
from pathlib import Path

from amber.common.config import ConfigLoader
from amber.common.logging import setup_logging
from amber.common.manifest import ArtifactManifest, new_run_id, write_manifest
from amber.exchange.bybit_client import BybitClient
from amber.storage.parquet_sink import ParquetSink
from amber.storage.state_store import StateStore

logger = logging.getLogger(__name__)


def main() -> None:
    config = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    setup_logging(config.get("run", {}).get("log_level", "INFO"))

    run_id = new_run_id(prefix="collector")
    data_root = Path(config["storage"]["raw_dir"])
    state_root = Path(config["storage"]["state_dir"])
    symbols = config["exchange"]["bybit"]["symbols"]

    sink = ParquetSink(data_root)
    state = StateStore(state_root)
    client = BybitClient()

    emitted = 0
    for symbol in symbols:
        tick = client.fetch_stub_tick(symbol)
        sink.write_records(topic="ticks", symbol=symbol, records=[client.as_dict(tick)])
        emitted += 1

    state.set("collector", {"run_id": run_id, "emitted_records": emitted})
    manifest = ArtifactManifest(
        run_id=run_id,
        artifact_type="collector_run",
        artifact_version="v1",
        created_at=client.fetch_stub_tick(symbols[0]).ts,
        config_ref="config/amber.yaml",
        feature_spec_ref="config/features.yaml",
        metadata={"symbols": symbols, "emitted_records": emitted},
    )
    write_manifest(data_root / "manifests" / run_id / "manifest.json", manifest)
    logger.info("collector finished run_id=%s emitted=%s", run_id, emitted)


if __name__ == "__main__":
    main()
