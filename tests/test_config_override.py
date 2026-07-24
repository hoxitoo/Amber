"""Tests for the local config override merge (prevents pull conflicts)."""

import tempfile
import unittest
from pathlib import Path

from amber.common.config import ConfigLoader, _deep_merge
from amber.dashboard.control import set_symbols


class TestConfigOverride(unittest.TestCase):
    def test_local_override_merges_and_replaces_lists(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config").mkdir()
            (root / "config" / "amber.yaml").write_text(
                "exchange:\n  bybit:\n    testnet: false\n    symbols:\n    - BTCUSDT\n"
                "storage:\n  keep_runs: 5\n",
                encoding="utf-8",
            )
            (root / "config" / "amber.local.yaml").write_text(
                "exchange:\n  bybit:\n    symbols:\n    - SOLUSDT\n    - XRPUSDT\n",
                encoding="utf-8",
            )
            cfg = ConfigLoader(root).load_yaml("config/amber.yaml")
            # list replaced by override, other keys preserved
            self.assertEqual(cfg["exchange"]["bybit"]["symbols"], ["SOLUSDT", "XRPUSDT"])
            self.assertEqual(cfg["exchange"]["bybit"]["testnet"], False)
            self.assertEqual(cfg["storage"]["keep_runs"], 5)

    def test_no_override_returns_base(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config").mkdir()
            (root / "config" / "amber.yaml").write_text("storage:\n  keep_runs: 3\n", encoding="utf-8")
            cfg = ConfigLoader(root).load_yaml("config/amber.yaml")
            self.assertEqual(cfg["storage"]["keep_runs"], 3)

    def test_set_symbols_writes_local_not_tracked(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config").mkdir()
            tracked = root / "config" / "amber.yaml"
            tracked.write_text("exchange:\n  bybit:\n    symbols:\n    - BTCUSDT\n", encoding="utf-8")
            before = tracked.read_text(encoding="utf-8")

            saved = set_symbols(root, ["ethusdt", " solusdt "])
            self.assertEqual(saved, ["ETHUSDT", "SOLUSDT"])
            # tracked file untouched; override created and applied on load
            self.assertEqual(tracked.read_text(encoding="utf-8"), before)
            self.assertTrue((root / "config" / "amber.local.yaml").exists())
            cfg = ConfigLoader(root).load_yaml("config/amber.yaml")
            self.assertEqual(cfg["exchange"]["bybit"]["symbols"], ["ETHUSDT", "SOLUSDT"])

    def test_deep_merge_recurses(self):
        base = {"a": {"b": 1, "c": 2}, "x": 1}
        _deep_merge(base, {"a": {"c": 9}, "y": 2})
        self.assertEqual(base, {"a": {"b": 1, "c": 9}, "x": 1, "y": 2})


if __name__ == "__main__":
    unittest.main()
