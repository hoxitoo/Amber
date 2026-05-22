from __future__ import annotations

import json
from pathlib import Path
<<<<<<< HEAD
import re
=======
>>>>>>> origin/main
from typing import Any


class StateStore:
    def __init__(self, state_root: Path) -> None:
        self.state_root = state_root
        self.state_root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
<<<<<<< HEAD
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
            raise ValueError(f"Invalid state key: {key!r}")
        path = (self.state_root / f"{key}.json").resolve()
        root = self.state_root.resolve()
        if root not in path.parents and path != root:
            raise ValueError("State path escapes state root")
        return path
=======
        return self.state_root / f"{key}.json"
>>>>>>> origin/main

    def get(self, key: str) -> dict[str, Any]:
        path = self._path(key)
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def set(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False, indent=2)
