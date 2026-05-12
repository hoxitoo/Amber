from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class StateStore:
    def __init__(self, state_root: Path) -> None:
        self.state_root = state_root
        self.state_root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.state_root / f"{key}.json"

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
