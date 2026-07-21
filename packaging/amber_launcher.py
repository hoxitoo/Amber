"""Entry point for a packaged (PyInstaller) Amber build.

Launches the Streamlit dashboard from inside a frozen executable. Building a
single-file exe of a Streamlit app is finicky; see packaging/README.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _project_root() -> Path:
    # When frozen, resources are unpacked next to the executable / in _MEIPASS.
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = _project_root()
    os.chdir(root)
    app = root / "amber" / "dashboard" / "app.py"
    sys.argv = ["streamlit", "run", str(app), "--global.developmentMode=false"]
    from streamlit.web import cli as stcli

    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
