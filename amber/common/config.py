from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def load_yaml(self, relative_path: str) -> dict[str, Any]:
        path = (self.root / relative_path).resolve()
        if self.root not in path.parents and path != self.root:
            raise ValueError(f"Config path escapes project root: {relative_path}")

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"Config {relative_path} must be a mapping")
        return data
