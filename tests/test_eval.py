import json
import tempfile
import unittest
from pathlib import Path

from amber.models.eval import evaluate_model


class TestEval(unittest.TestCase):
    def _make_model(self, root: Path) -> Path:
        models = root / "models"
        model_dir = models / "model_1"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.json").write_text(
            json.dumps({"weights": {"ret_1": 0.1, "vol_z_20": 0.1}, "bias": 0.0, "model_type": "baseline"}),
            encoding="utf-8",
        )
        return models

    def test_evaluate_model_rejects_bad_threshold(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            models = self._make_model(root)
            with self.assertRaisesRegex(ValueError, "threshold must be in \\[0, 1\\]"):
                evaluate_model(models_root=models, datasets_root=root / "datasets", threshold=1.5)

    def test_evaluate_model_raises_on_invalid_dataset_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            models = self._make_model(root)
            ds = root / "datasets" / "dataset_1"
            ds.mkdir(parents=True)
            (ds / "dataset.jsonl").write_text('{"up_hit": 1}\n{bad json}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid JSONL"):
                evaluate_model(models_root=models, datasets_root=root / "datasets", threshold=0.7)

    def test_evaluate_model_returns_dual_head_calibrated_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            models = root / "models"
            model_dir = models / "model_1"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "model.json").write_text(
                json.dumps(
                    {
                        "heads": {
                            "pump": {"weights": {"ret_1": 0.2, "vol_z_20": 0.1}, "bias": 0.0},
                            "dump": {"weights": {"ret_1": -0.2, "vol_z_20": 0.1}, "bias": 0.0},
                        },
                        "model_type": "baseline_dual",
                    }
                ),
                encoding="utf-8",
            )
            calib_dir = models / "calib_1"
            calib_dir.mkdir(parents=True)
            (calib_dir / "calibration.json").write_text(
                json.dumps(
                    {
                        "method": "multi_head",
                        "heads": {
                            "pump": {"method": "scalar_mean_calibration", "scale": 1.1},
                            "dump": {"method": "scalar_mean_calibration", "scale": 0.9},
                        },
                    }
                ),
                encoding="utf-8",
            )
            ds = root / "datasets" / "dataset_1"
            ds.mkdir(parents=True)
            with (ds / "dataset.jsonl").open("w", encoding="utf-8") as fh:
                for i in range(60):
                    fh.write(json.dumps({"ret_1": 0.001 * (i % 5), "vol_z_20": 1.0, "up_hit": i % 2, "down_hit": (i + 1) % 2}) + "\n")

            out = evaluate_model(models_root=models, datasets_root=root / "datasets", threshold=0.5)
            self.assertEqual(out["rows"], 60.0)
            self.assertEqual(out["calibration_method"], "multi_head")
            self.assertIn("avg_prob_down", out)
            self.assertIn("brier_up_cal", out)
            self.assertIn("brier_down_cal", out)
            self.assertIn("precision_up_at_threshold", out)
            self.assertIn("precision_down_at_threshold", out)
            self.assertEqual(out["precision_at_threshold"], out["precision_up_at_threshold"])


if __name__ == "__main__":
    unittest.main()
