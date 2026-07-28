"""Reclaim disk space: prune old run artifacts and consumed raw WS files.

Run from the dashboard control panel or: python scripts/cleanup.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
from amber.common.retention import cleanup_consumed_ws_raw, dataset_keep, prune_run_dirs
from amber.pipeline.normalize_app import OFFSETS_STATE_KEY
from amber.storage.state_store import StateStore


def main() -> None:
    cfg = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    storage = cfg["storage"]
    keep = int(storage.get("keep_runs", 5))

    datasets_root = Path(storage["datasets_dir"])
    models_root = Path(storage["models_dir"])
    raw_root = Path(storage["raw_dir"])
    state = StateStore(Path(storage["state_dir"]))

    freed = 0
    deleted = 0
    for root, prefix, n in (
        (datasets_root, "dataset_", dataset_keep(storage)),
        (models_root, "model_", keep),
        (models_root, "calib_", keep),
    ):
        d, f = prune_run_dirs(root, prefix, n)
        deleted += d
        freed += f

    offsets = dict(state.get(OFFSETS_STATE_KEY))
    fd, ff = cleanup_consumed_ws_raw(raw_root, offsets)
    state.set(OFFSETS_STATE_KEY, offsets)

    print(
        f"Очистка завершена: удалено {deleted} старых прогонов + {fd} потреблённых ws_raw файлов, "
        f"освобождено ~{(freed + ff) / 1e6:.1f} МБ (храним последние {keep} прогонов)."
    )


if __name__ == "__main__":
    main()
