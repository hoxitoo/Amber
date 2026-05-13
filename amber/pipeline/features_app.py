from __future__ import annotations

import logging
from pathlib import Path

from amber.common.config import ConfigLoader
from amber.common.logging import setup_logging
from amber.common.manifest import ArtifactManifest, new_run_id, write_manifest
from amber.features.compute import compute_batch_features
from amber.storage.state_store import StateStore

logger = logging.getLogger(__name__)


def main() -> None:
    config = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    setup_logging(config.get("run", {}).get("log_level", "INFO"))

    run_id = new_run_id(prefix="features")
    raw_root = Path(config["storage"]["raw_dir"])
    out_root = Path(config["storage"]["features_dir"])
    state_root = Path(config["storage"]["state_dir"])
    symbols = config["exchange"]["bybit"]["symbols"]

    result = compute_batch_features(raw_root=raw_root, out_root=out_root, symbols=symbols)

    state = StateStore(state_root)
    state.set("features", {"run_id": run_id, **result})

    manifest = ArtifactManifest(
        run_id=run_id,
        artifact_type="features_run",
        artifact_version="v1",
        created_at=run_id.split("_")[1],
        config_ref="config/amber.yaml",
        feature_spec_ref="config/features.yaml",
        metadata={"symbols": symbols, **result},
    )
    write_manifest(out_root / "manifests" / run_id / "manifest.json", manifest)

    logger.info("features finished run_id=%s written_rows=%s", run_id, result["written_rows"])


if __name__ == "__main__":
    main()
