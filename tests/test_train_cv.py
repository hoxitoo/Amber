import json
import tempfile
import unittest
from pathlib import Path

from amber.models.train import train_model


class TestTrainCVConfig(unittest.TestCase):
    def test_train_model_respects_configured_cv_gap(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            datasets_root = root / "datasets"
            models_root = root / "models"
            ds = datasets_root / "dataset_test"
            ds.mkdir(parents=True)
            rows = []
            for i in range(200):
                rows.append({"ret_1": 0.001 * (i % 5), "vol_z_20": 1.0, "up_hit": i % 2, "down_hit": (i + 1) % 2})
            with (ds / "dataset.jsonl").open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")

            out = train_model(datasets_root=datasets_root, models_root=models_root, cv_n_folds=3, cv_val_size=40, cv_gap_candles=30)
            model_path = models_root / out["run_id"] / "model.json"
            manifest_path = models_root / out["run_id"] / "manifest.json"
            model = json.loads(model_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(out["cv_gap_candles"], 30)
            self.assertEqual(out["cv_generated_folds"], 3)
            self.assertEqual(out["cv_skipped_folds"], 0)
            self.assertEqual(model["cv"]["gap_candles"], 30)
            self.assertEqual(model["cv"]["val_size"], 40)
            self.assertEqual(model["cv"]["generated_folds"], 3)
            self.assertEqual(model["cv"]["skipped_folds"], 0)
            self.assertIn("brier_mean", model["cv"])
            self.assertIn("brier_std", model["cv"])
            self.assertEqual(manifest["metadata"]["cv_folds"], out["cv_folds"])
            self.assertEqual(manifest["metadata"]["cv_generated_folds"], out["cv_generated_folds"])
            self.assertEqual(manifest["metadata"]["cv_skipped_folds"], out["cv_skipped_folds"])
            self.assertEqual(manifest["metadata"]["cv_gap_candles"], out["cv_gap_candles"])
            self.assertEqual(manifest["metadata"]["cv_brier_mean"], out["cv_brier_mean"])


if __name__ == "__main__":
    unittest.main()
