"""Rolling recalibration must look after the head the scanner gates on.

Until 2026-09-29 it checked only pump and dump, and a refit wrote a
calibration holding those two heads alone. The first refit after a retrain
therefore dropped the `move` head's calibration, and the gate ran on raw
scores — a different alert rate, with no log line — until the next retrain.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from amber.models.recalibrate import check_and_recalibrate
from amber.signals.scorer import _load_latest_calibration

WELL_CALIBRATED = {"method": "platt", "a": 1.0, "b": 0.0}
# Pushes every probability far above the observed rate: a guaranteed refit.
OVERCONFIDENT = {"method": "platt", "a": 1.0, "b": 3.0}


def _setup(tmp: Path, move_cal: dict, pump_cal: dict, seed: int = 0) -> tuple[Path, Path]:
    rng = random.Random(seed)
    models, datasets = tmp / "models", tmp / "datasets"
    run = models / "model_20260101T000000Z"
    run.mkdir(parents=True)
    heads = {
        name: {"type": "logreg", "weights": {"vol_z_20": 1.5}, "bias": -1.0, "label_rate": 0.3}
        for name in ("move", "pump", "dump")
    }
    (run / "model.json").write_text(json.dumps({
        "model_type": "logreg_dual_v1", "features": ["vol_z_20"], "heads": heads,
    }))
    (models / "registry.json").write_text(json.dumps({"model_run_id": run.name}))
    cal = models / "calib_20260101T000001Z"
    cal.mkdir()
    (cal / "calibration.json").write_text(json.dumps({
        "method": "multi_head", "model_run_id": run.name,
        "heads": {"move": move_cal, "pump": pump_cal, "dump": WELL_CALIBRATED},
    }))
    ds = datasets / "dataset_20260101T000000Z"
    ds.mkdir(parents=True)
    with (ds / "dataset.jsonl").open("w") as fh:
        for i in range(3000):
            x = rng.gauss(0, 1)
            # Labels drawn from the model's own probability: its raw score is
            # calibrated, so identity-like Platt is right and b=+3 is wrong.
            p = 1 / (1 + pow(2.718281828, -(1.5 * x - 1.0)))
            hit = int(rng.random() < p)
            fh.write(json.dumps({
                "vol_z_20": x, "ts": i * 60_000, "symbol": "AAAUSDT", "horizon_steps": 15,
                "up_hit": hit, "down_hit": hit, "move_hit": hit,
            }) + "\n")
    return models, datasets


class TestRecalibrationKeepsTheGatingHead(unittest.TestCase):
    def test_refitting_another_head_keeps_the_move_calibration(self):
        with tempfile.TemporaryDirectory() as td:
            models, datasets = _setup(Path(td), move_cal=WELL_CALIBRATED, pump_cal=OVERCONFIDENT)
            res = check_and_recalibrate(models, datasets)
            self.assertTrue(res["refit"], res)
            self.assertTrue(res["heads"]["pump"]["refit"])
            latest = _load_latest_calibration(models)
            self.assertEqual(latest["heads"].get("move"), WELL_CALIBRATED, "the move calibration was dropped")
            self.assertEqual(latest["heads"].get("dump"), WELL_CALIBRATED)

    def test_the_move_head_itself_is_checked_and_refit(self):
        with tempfile.TemporaryDirectory() as td:
            models, datasets = _setup(Path(td), move_cal=OVERCONFIDENT, pump_cal=WELL_CALIBRATED)
            res = check_and_recalibrate(models, datasets)
            self.assertIn("move", res["heads"], "the gating head was never examined")
            self.assertTrue(res["heads"]["move"]["refit"])
            latest = _load_latest_calibration(models)
            self.assertNotEqual(latest["heads"]["move"], OVERCONFIDENT)
            self.assertEqual(latest["heads"]["pump"], WELL_CALIBRATED)


if __name__ == "__main__":
    unittest.main()
