import unittest
<<<<<<< HEAD
import tempfile
import json
from pathlib import Path

from amber.models.calibrate import calibrate_model
=======

>>>>>>> origin/main
from amber.signals.scorer import calibrated_prob


class TestCalibration(unittest.TestCase):
    def test_isotonic_interpolation(self):
        calib = {"method": "isotonic", "x_thresholds": [0.0, 0.5, 1.0], "y_thresholds": [0.0, 0.6, 1.0]}
        p = calibrated_prob(0.25, calib)
        self.assertGreater(p, 0.25)
        self.assertLessEqual(p, 1.0)

<<<<<<< HEAD
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

=======
>>>>>>> origin/main

if __name__ == "__main__":
    unittest.main()
