"""The scanner must idle gracefully (not crash) before any model is trained,
even when data/models does not exist yet (the Windows FileNotFoundError case)."""

import tempfile
import unittest
from pathlib import Path

from amber.alerts.router import AlertRateLimiter
from amber.pipeline.scanner_app import scan_once
from amber.signals.filters import SignalGate


class TestScannerNoModel(unittest.TestCase):
    def _config(self, root: Path) -> dict:
        return {
            "storage": {
                "features_dir": str(root / "features"),
                "models_dir": str(root / "models"),  # intentionally does not exist
                "logs_dir": str(root / "logs"),
                "state_dir": str(root / "state"),
            },
            "signal": {"schema_version": "v1", "top_k_universe": 20},
            "labeling": {"min_warmup_bars": 60},
        }

    def test_scan_once_returns_zero_without_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # a feature file exists so the universe step runs, but no model dir
            feat = root / "features" / "features" / "BTCUSDT"
            feat.mkdir(parents=True)
            (feat / "part-000.jsonl").write_text('{"symbol":"BTCUSDT","ts":1,"obs":100}\n', encoding="utf-8")
            cfg = self._config(root)
            gate = SignalGate(cooldown_sec=0, concurrent_limit=5)
            emitted = scan_once(cfg, {"pump_prob_calibrated_min": 0.5, "dump_prob_calibrated_min": 0.5},
                                gate, AlertRateLimiter(cooldown_sec=0))
            self.assertEqual(emitted, 0)


if __name__ == "__main__":
    unittest.main()
