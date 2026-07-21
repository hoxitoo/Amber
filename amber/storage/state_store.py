from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


class StateStore:
    def __init__(self, state_root: Path) -> None:
        self.state_root = state_root
        self.state_root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
            raise ValueError(f"Invalid state key: {key!r}")
        path = (self.state_root / f"{key}.json").resolve()
        root = self.state_root.resolve()
        if root not in path.parents and path != root:
            raise ValueError("State path escapes state root")
        return path

    def get(self, key: str) -> dict[str, Any]:
        path = self._path(key)
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Persist state atomically: write to a temp file in the same directory,
        fsync, then os.replace (atomic rename) so a crash mid-write can never
        leave a truncated/corrupt watermark or offset file."""
        path = self._path(key)
        fd, tmp_name = tempfile.mkstemp(dir=str(self.state_root), prefix=f".{key}.", suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(value, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)  # atomic on POSIX and Windows (same filesystem)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
