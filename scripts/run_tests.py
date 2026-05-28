from pathlib import Path
import importlib.util
import os
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED_IMPORTS = ("pydantic", "yaml", "sklearn", "websockets")


def _missing_deps() -> list[str]:
    return [name for name in REQUIRED_IMPORTS if importlib.util.find_spec(name) is None]


def _ensure_dependencies() -> None:
    missing = _missing_deps()
    if not missing:
        return

    auto_install = os.environ.get("AMBER_AUTO_INSTALL_DEPS", "0") == "1"
    if not auto_install:
        raise SystemExit(
            "Missing dependencies: "
            + ", ".join(missing)
            + ". Install with: python -m pip install -r requirements.txt. Or set AMBER_AUTO_INSTALL_DEPS=1 to auto-install."
        )

    req = ROOT / "requirements.txt"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])


if __name__ == "__main__":
    _ensure_dependencies()
    suite = unittest.defaultTestLoader.discover("tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
