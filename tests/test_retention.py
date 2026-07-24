"""Tests for disk retention: pruning old runs and consumed ws_raw cleanup."""

import tempfile
import unittest
from pathlib import Path

from amber.common.retention import cleanup_consumed_ws_raw, prune_run_dirs


class TestPruneRuns(unittest.TestCase):
    def test_keeps_newest_and_deletes_rest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("dataset_20260101", "dataset_20260102", "dataset_20260103", "other"):
                (root / name).mkdir()
                (root / name / "f.txt").write_text("x" * 100, encoding="utf-8")
            deleted, freed = prune_run_dirs(root, "dataset_", keep=2)
            self.assertEqual(deleted, 1)
            self.assertGreater(freed, 0)
            remaining = sorted(p.name for p in root.iterdir())
            self.assertEqual(remaining, ["dataset_20260102", "dataset_20260103", "other"])

    def test_keep_zero_is_noop(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "dataset_1").mkdir()
            self.assertEqual(prune_run_dirs(root, "dataset_", keep=0), (0, 0))


class TestCleanupWsRaw(unittest.TestCase):
    def test_deletes_consumed_old_files_only(self):
        with tempfile.TemporaryDirectory() as td:
            raw_root = Path(td) / "raw"
            sym = raw_root / "ws_raw" / "BTCUSDT"
            sym.mkdir(parents=True)
            old = sym / "part-2026010100.jsonl"
            new = sym / "part-2026010101.jsonl"
            old.write_text("a\nb\n", encoding="utf-8")  # 4 bytes
            new.write_text("c\nd\n", encoding="utf-8")
            offsets = {
                "BTCUSDT/part-2026010100.jsonl": old.stat().st_size,  # fully consumed
                "BTCUSDT/part-2026010101.jsonl": 0,  # newest, keep
            }
            deleted, freed = cleanup_consumed_ws_raw(raw_root, offsets)
            self.assertEqual(deleted, 1)
            self.assertFalse(old.exists())
            self.assertTrue(new.exists())
            self.assertNotIn("BTCUSDT/part-2026010100.jsonl", offsets)

    def test_keeps_unconsumed_file(self):
        with tempfile.TemporaryDirectory() as td:
            raw_root = Path(td) / "raw"
            sym = raw_root / "ws_raw" / "BTCUSDT"
            sym.mkdir(parents=True)
            old = sym / "part-2026010100.jsonl"
            new = sym / "part-2026010101.jsonl"
            old.write_text("a\nb\nc\n", encoding="utf-8")
            new.write_text("d\n", encoding="utf-8")
            offsets = {"BTCUSDT/part-2026010100.jsonl": 2}  # only partially consumed
            deleted, _ = cleanup_consumed_ws_raw(raw_root, offsets)
            self.assertEqual(deleted, 0)
            self.assertTrue(old.exists())


if __name__ == "__main__":
    unittest.main()
