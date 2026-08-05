"""Every consumer of `train_reference` must handle the format the model writes.

The reference changed from a bare list of quantile edges to
{"edges", "expected"} (or {"values", "expected"} for discrete features).
quality_report kept doing list(edges), which on a dict yields its KEYS, so the
system report died with `could not convert string to float: 'edges'` and the
whole dashboard showed "Нет данных для отчёта". These tests exercise each
consumer against a real model artifact rather than a hand-made list.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from amber.models.train import _feature_quantiles
from amber.monitoring.drift import detect_drift, psi_from_quantile_reference
from amber.monitoring.quality_report import _feature_psi

FEATURES = ("ret_1", "vol_z_20", "spread_bps", "breakout_up_20")


def _train_rows(n: int = 3000) -> list[dict]:
    rng = random.Random(4)
    return [
        {
            "ret_1": rng.gauss(0, 0.01),
            "vol_z_20": rng.gauss(0, 1),
            "spread_bps": abs(rng.gauss(2, 0.5)),
            "breakout_up_20": 1.0 if i % 20 == 0 else 0.0,
        }
        for i in range(n)
    ]


def _write_model(models_root: Path, reference: dict) -> None:
    d = models_root / "model_20260101T000000Z_abcd1234"
    d.mkdir(parents=True)
    (d / "model.json").write_text(
        json.dumps({"model_type": "lightgbm_dual_v1", "heads": {}, "train_reference": reference}),
        encoding="utf-8",
    )


def _write_features(features_root: Path, n: int = 600) -> None:
    rng = random.Random(9)
    d = features_root / "features" / "BTCUSDT"
    d.mkdir(parents=True)
    with (d / "part-000.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({
                "ts": 1_700_000_000_000 + i * 60_000,
                "ret_1": rng.gauss(0, 0.01),
                "vol_z_20": rng.gauss(0, 1),
                "spread_bps": abs(rng.gauss(2, 0.5)),
                "breakout_up_20": 1.0 if i % 20 == 0 else 0.0,
            }) + "\n")


class TestReferenceConsumers(unittest.TestCase):
    def test_quality_report_handles_the_model_format(self):
        """This is the exact path that crashed the dashboard."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_model(root / "models", _feature_quantiles(_train_rows()))
            _write_features(root / "features")

            res = _feature_psi(root / "models", root / "features", window=500)
            self.assertEqual(res["reason"], "ok", res)
            self.assertIn("ret_1", res["per_feature"])
            for name, psi in res["per_feature"].items():
                self.assertIsInstance(psi, float)
                self.assertLess(psi, 0.2, f"{name} drifted against identical data")

    def test_detect_drift_handles_the_model_format(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_features(root / "features")
            ref = _feature_quantiles(_train_rows())
            res = detect_drift(root / "features", "BTCUSDT", reference=ref)
            self.assertEqual(res["reference"], "train_quantiles")
            self.assertLess(res["max_psi"], 0.2)

    def test_legacy_bare_edge_lists_still_work(self):
        """Models trained before the change must not crash the report."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            legacy = {"ret_1": [-0.03 + i * 0.006 for i in range(11)]}
            _write_model(root / "models", legacy)
            _write_features(root / "features")
            res = _feature_psi(root / "models", root / "features", window=500)
            self.assertEqual(res["reason"], "ok", res)
            self.assertIsInstance(res["per_feature"]["ret_1"], float)

    def test_reference_survives_a_json_round_trip(self):
        """The model artifact is JSON on disk, so the format must reload intact."""
        ref = json.loads(json.dumps(_feature_quantiles(_train_rows())))
        live = [random.Random(1).gauss(0, 0.01) for _ in range(300)]
        self.assertIsInstance(psi_from_quantile_reference(ref["ret_1"], live), float)
        self.assertIn("expected", ref["ret_1"])


if __name__ == "__main__":
    unittest.main()
