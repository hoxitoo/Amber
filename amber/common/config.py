from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def load_yaml(self, relative_path: str) -> dict[str, Any]:
<<<<<<< HEAD
        if not relative_path.endswith((".yaml", ".yml")):
            raise ValueError(f"Config must be YAML (.yaml/.yml): {relative_path}")
=======
>>>>>>> origin/main
        path = (self.root / relative_path).resolve()
        if self.root not in path.parents and path != self.root:
            raise ValueError(f"Config path escapes project root: {relative_path}")

<<<<<<< HEAD
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {relative_path}")

=======
>>>>>>> origin/main
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"Config {relative_path} must be a mapping")
        return data
