import tempfile
import unittest
from pathlib import Path

from amber.common.paths import artifact_dir
from amber.storage.state_store import StateStore


class TestPathSafety(unittest.TestCase):
    def test_artifact_dir_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(ValueError):
                artifact_dir(root, "..", "run_1")
            with self.assertRaises(ValueError):
                artifact_dir(root, "models", "../run_1")

    def test_state_store_rejects_invalid_key(self):
        with tempfile.TemporaryDirectory() as td:
            store = StateStore(Path(td))
            with self.assertRaises(ValueError):
                store.set("../secret", {"a": 1})
            with self.assertRaises(ValueError):
                _ = store.get("nested/key")


if __name__ == "__main__":
    unittest.main()
