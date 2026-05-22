import tempfile
import unittest
from pathlib import Path

from amber.common.config import ConfigLoader


class TestConfigLoader(unittest.TestCase):
    def test_load_yaml_rejects_non_yaml_extension(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "amber.txt").write_text("a: 1\n", encoding="utf-8")
            loader = ConfigLoader(root)
            with self.assertRaisesRegex(ValueError, "Config must be YAML"):
                loader.load_yaml("config/amber.txt")

    def test_load_yaml_raises_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            loader = ConfigLoader(Path(td))
            with self.assertRaises(FileNotFoundError):
                loader.load_yaml("config/missing.yaml")


if __name__ == "__main__":
    unittest.main()
