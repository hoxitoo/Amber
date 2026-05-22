import tempfile
import unittest
from pathlib import Path

from amber.pipeline.normalize_app import _iter_jsonl_payloads


class TestNormalizeApp(unittest.TestCase):
    def test_iter_jsonl_payloads_skips_invalid_lines(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "part-000.jsonl"
            path.write_text('{"a":1}\n{bad json\n\n{"b":2}\n', encoding='utf-8')
            rows = list(_iter_jsonl_payloads(path))
            self.assertEqual(rows, [{"a": 1}, {"b": 2}])


if __name__ == "__main__":
    unittest.main()
