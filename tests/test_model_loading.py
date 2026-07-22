import json
import tempfile
import unittest
from pathlib import Path

from amber.models.infer import load_latest_model
from amber.models.registry import latest_registered, register_model
from amber.signals.scorer import score_signal


class TestModelLoading(unittest.TestCase):
    def test_load_latest_model_raises_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "No model_\\* directories found"):
                load_latest_model(Path(td))

    def test_load_latest_model_raises_valueerror_when_dir_absent(self):
        # models_root does not exist at all (fresh install before any training):
        # must raise ValueError, not FileNotFoundError, so the scanner can catch it.
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "data" / "models"
            with self.assertRaises(ValueError):
                load_latest_model(missing)

    def test_score_signal_falls_back_when_calibration_broken(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "model_1"
            model_dir.mkdir(parents=True)
            (model_dir / "model.json").write_text(
                json.dumps({"heads": {"pump": {"weights": {"ret_1": 0.5, "vol_z_20": -0.2}, "bias": 0.0}, "dump": {"weights": {"ret_1": -0.5, "vol_z_20": -0.2}, "bias": 0.0}}, "model_type": "baseline_dual"}),
                encoding="utf-8",
            )
            calib_dir = root / "calib_1"
            calib_dir.mkdir(parents=True)
            (calib_dir / "calibration.json").write_text("{bad json", encoding="utf-8")

            sig = score_signal({"ts": 1, "symbol": "BTCUSDT", "ret_1": 0.01, "vol_z_20": 0.1}, root)
            self.assertGreaterEqual(sig.prob_up_calibrated, 0.0)
            self.assertLessEqual(sig.prob_up_calibrated, 1.0)

    def test_latest_registered_skips_corrupted_lines(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            register_model(root, model_run_id="model_1")
            (root / "registry.jsonl").write_text('{"bad_json"\n', encoding="utf-8")
            # keep at least one valid line after the broken one
            with (root / "registry.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"model_run_id": "model_2"}) + "\n")
            last = latest_registered(root)
            self.assertIsNotNone(last)
            self.assertEqual(last["model_run_id"], "model_2")

    def test_score_signal_contains_dual_head_impacts_and_directional_rule(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "model_1"
            model_dir.mkdir(parents=True)
            (model_dir / "model.json").write_text(
                json.dumps({"heads": {"pump": {"weights": {"ret_1": 0.5, "vol_z_20": -0.2}, "bias": 0.0}, "dump": {"weights": {"ret_1": -0.3, "vol_z_20": 0.1}, "bias": 0.0}}, "model_type": "baseline_dual"}),
                encoding="utf-8",
            )
            sig = score_signal({"ts": 1, "symbol": "BTCUSDT", "ret_1": 0.01, "vol_z_20": 0.1}, root)
            keys = {list(x.keys())[0] for x in sig.explanation.top_feature_impacts}
            self.assertEqual(len(sig.explanation.top_feature_impacts), 3)
            self.assertIn("pump_ret_1", keys)
            self.assertIn("dump_vol_z_20", keys)
            rules = [r.get("rule") for r in sig.explanation.rule_trace]
            self.assertIn("explanation_method", rules)
            self.assertIn("directional_score", rules)



if __name__ == "__main__":
    unittest.main()
