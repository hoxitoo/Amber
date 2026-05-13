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


if __name__ == "__main__":
    unittest.main()
