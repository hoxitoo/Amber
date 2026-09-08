from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def enter_project_root(script_file: str | Path) -> Path:
    """chdir to the project root and return it.

    Storage paths in `config/amber.yaml` are relative (`data/raw`, ...), and the
    systemd units resolve them by setting `WorkingDirectory=/opt/amber`. A
    script run from anywhere else therefore points at directories that do not
    exist — or, run from a home directory, fails to open its own file with a
    permission error that says nothing about the actual mistake. Anchoring on
    the script's own location makes an analysis script behave identically
    wherever it is invoked from.
    """
    root = Path(script_file).resolve().parents[1]
    os.chdir(root)
    return root


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base`. Dicts merge; other values
    (including lists like the symbol universe) replace."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


class ConfigLoader:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _load_one(self, path: Path, relative_path: str) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"Config {relative_path} must be a mapping")
        return data

    def load_yaml(self, relative_path: str) -> dict[str, Any]:
        if not relative_path.endswith((".yaml", ".yml")):
            raise ValueError(f"Config must be YAML (.yaml/.yml): {relative_path}")
        path = (self.root / relative_path).resolve()
        if self.root not in path.parents and path != self.root:
            raise ValueError(f"Config path escapes project root: {relative_path}")

        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {relative_path}")

        data = self._load_one(path, relative_path)

        # Local override: `<name>.local.yaml` (gitignored) is merged on top so a
        # user's runtime edits (e.g. the symbol list) never conflict with pulls
        # of the tracked defaults.
        local = path.with_name(f"{path.stem}.local{path.suffix}")
        if local.is_file():
            _deep_merge(data, self._load_one(local, local.name))
        return data
