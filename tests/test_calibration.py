import unittest
import tempfile
import json
from pathlib import Path

from amber.models.calibrate import calibrate_model
from amber.signals.scorer import calibrated_prob, calibrated_prob_for_target


class TestCalibration(unittest.TestCase):
    def test_isotonic_interpolation(self):
        calib = {"method": "isotonic", "x_thresholds": [0.0, 0.5, 1.0], "y_thresholds": [0.0, 0.6, 1.0]}
        p = calibrated_prob(0.25, calib)
        self.assertGreater(p, 0.25)
        self.assertLessEqual(p, 1.0)

    def test_calibrate_model_raises_if_no_datasets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            models = root / "models"
            models.mkdir(parents=True)
            m = models / "model_1"
            m.mkdir(parents=True)
            (m / "model.json").write_text(
                json.dumps({"heads": {"pump": {"weights": {"ret_1": 0.1, "vol_z_20": 0.1}, "bias": 0.0}, "dump": {"weights": {"ret_1": -0.1, "vol_z_20": 0.1}, "bias": 0.0}}, "model_type": "baseline_dual"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "No dataset_\\* directories found"):
                calibrate_model(models_root=models, datasets_root=root / "datasets")

    def test_calibrate_model_raises_on_invalid_dataset_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            models = root / "models"
            models.mkdir(parents=True)
            m = models / "model_1"
            m.mkdir(parents=True)
            (m / "model.json").write_text(
                json.dumps({"heads": {"pump": {"weights": {"ret_1": 0.1, "vol_z_20": 0.1}, "bias": 0.0}, "dump": {"weights": {"ret_1": -0.1, "vol_z_20": 0.1}, "bias": 0.0}}, "model_type": "baseline_dual"}),
                encoding="utf-8",
            )
            ds = root / "datasets" / "dataset_1"
            ds.mkdir(parents=True)
            (ds / "dataset.jsonl").write_text('{"up_hit": 1}\n{bad json}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid JSONL"):
                calibrate_model(models_root=models, datasets_root=root / "datasets")

    def test_calibrated_prob_for_target_uses_head_specific_payload(self):
        calib = {
            "method": "multi_head",
            "heads": {
                "pump": {"method": "scalar_mean_calibration", "scale": 2.0},
                "dump": {"method": "scalar_mean_calibration", "scale": 0.5},
            },
        }
        self.assertAlmostEqual(calibrated_prob_for_target(0.2, calib, target="pump"), 0.4)
        self.assertAlmostEqual(calibrated_prob_for_target(0.2, calib, target="dump"), 0.1)

    def test_calibrate_model_writes_multi_head_payload(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            models = root / "models"
            models.mkdir(parents=True)
            m = models / "model_1"
            m.mkdir(parents=True)
            (m / "model.json").write_text(
                json.dumps({"heads": {"pump": {"weights": {"ret_1": 0.2, "vol_z_20": 0.1}, "bias": 0.0}, "dump": {"weights": {"ret_1": -0.2, "vol_z_20": 0.1}, "bias": 0.0}}, "model_type": "baseline_dual"}),
                encoding="utf-8",
            )
            ds = root / "datasets" / "dataset_1"
            ds.mkdir(parents=True)
            with (ds / "dataset.jsonl").open("w", encoding="utf-8") as fh:
                for i in range(100):
                    fh.write(json.dumps({"ret_1": (i % 5) * 0.001, "vol_z_20": 1.0, "up_hit": i % 2, "down_hit": (i + 1) % 2}) + "\n")

            out = calibrate_model(models_root=models, datasets_root=root / "datasets", holdout_ratio=0.3)
            self.assertEqual(out["method"], "multi_head")
            self.assertEqual(out["holdout_rows"], 30)
            payload = json.loads((models / out["run_id"] / "calibration.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["method"], "multi_head")
            self.assertIn("pump", payload["heads"])
            self.assertIn("dump", payload["heads"])
            self.assertEqual(payload["holdout_ratio"], 0.3)
            self.assertEqual(payload["rows"], 30)

    def test_calibrate_model_rejects_bad_holdout_ratio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            models = root / "models"
            models.mkdir(parents=True)
            m = models / "model_1"
            m.mkdir(parents=True)
            (m / "model.json").write_text(
                json.dumps({"heads": {"pump": {"weights": {"ret_1": 0.2, "vol_z_20": 0.1}, "bias": 0.0}, "dump": {"weights": {"ret_1": -0.2, "vol_z_20": 0.1}, "bias": 0.0}}, "model_type": "baseline_dual"}),
                encoding="utf-8",
            )
            ds = root / "datasets" / "dataset_1"
            ds.mkdir(parents=True)
            (ds / "dataset.jsonl").write_text(json.dumps({"ret_1": 0.0, "vol_z_20": 1.0, "up_hit": 0, "down_hit": 1}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "holdout_ratio must be in \\(0, 1\\)"):
                calibrate_model(models_root=models, datasets_root=root / "datasets", holdout_ratio=1.0)


if __name__ == "__main__":
    unittest.main()
