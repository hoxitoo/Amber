import json
import tempfile
import unittest
from pathlib import Path

from amber.monitoring.quality_report import build_quality_report


class TestQualityReport(unittest.TestCase):
    def test_quality_report_ignores_bad_rows(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "signals.jsonl"
            rows = [
                json.dumps({"prob_up_calibrated": 0.8, "prob_down_calibrated": 0.2}),
                "{bad",
                json.dumps({"prob_up_calibrated": 0.6, "prob_down_calibrated": 0.4}),
            ]
            p.write_text("\n".join(rows) + "\n", encoding="utf-8")
            rep = build_quality_report(p)
            self.assertEqual(rep["signals"], 2)
            self.assertIn("psi", rep)




    def test_quality_report_sanitizes_non_finite_probs(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "signals.jsonl"
            rows = [
                json.dumps({"prob_up_calibrated": "nan", "prob_down_calibrated": 0.2}),
                json.dumps({"prob_up_calibrated": 1.2, "prob_down_calibrated": -0.1}),
            ]
            p.write_text("\n".join(rows) + "\n", encoding="utf-8")
            rep = build_quality_report(p)
            self.assertEqual(rep["signals"], 2)
            self.assertIsNone(rep["prediction_bias"])

if __name__ == "__main__":
    unittest.main()
