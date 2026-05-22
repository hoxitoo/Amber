import tempfile
import unittest
from pathlib import Path

from amber.monitoring.health import check_health


class TestHealth(unittest.TestCase):
    def test_check_health_reports_ages_and_ok(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw" / "ticks" / "BTCUSDT"
            feat = root / "features" / "features" / "BTCUSDT"
            model = root / "models" / "model_1"
            raw.mkdir(parents=True)
            feat.mkdir(parents=True)
            model.mkdir(parents=True)
            (raw / "part-000.jsonl").write_text("{}", encoding="utf-8")
            (feat / "part-000.jsonl").write_text("{}", encoding="utf-8")
            (model / "model.json").write_text("{}", encoding="utf-8")

            report = check_health(root, max_age_sec=3600)
            self.assertTrue(report.ok)
            self.assertIn("raw_age_sec", report.checks)
            self.assertIn("features_age_sec", report.checks)
            self.assertIn("model_age_sec", report.checks)




    def test_check_health_rejects_non_positive_max_age(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(ValueError, "max_age_sec must be positive"):
                check_health(root, max_age_sec=0)


if __name__ == "__main__":
    unittest.main()
