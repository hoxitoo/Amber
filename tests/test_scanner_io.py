import json
import tempfile
import unittest
from pathlib import Path

from amber.pipeline.scanner_app import _read_latest_feature_rows


class TestScannerIO(unittest.TestCase):
    def test_read_latest_feature_rows_skips_corrupted_tail(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "features" / "BTCUSDT"
            p.mkdir(parents=True)
            rows = [
                json.dumps({"symbol": "BTCUSDT", "ts": 1}),
                "{bad",
                json.dumps({"symbol": "BTCUSDT", "ts": 2}),
            ]
            (p / "part-000.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

            out = _read_latest_feature_rows(root)
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0]["ts"], 2)

    def test_read_latest_feature_rows_skips_fully_corrupted_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "features" / "ETHUSDT"
            p.mkdir(parents=True)
            (p / "part-000.jsonl").write_text("{bad\n{still bad\n", encoding="utf-8")

            out = _read_latest_feature_rows(root)
            self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
