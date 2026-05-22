import json
import unittest
import tempfile
from pathlib import Path

from amber.monitoring.drift import PredictionBiasMonitor, RollingAUCMonitor, detect_drift, psi_monitor


class TestQualityMonitors(unittest.TestCase):
    def test_bias_monitor(self):
        m = PredictionBiasMonitor(window=50)
        for _ in range(25):
            m.update(0.7, 0.3)
        self.assertIsNotNone(m.bias())
        self.assertGreater(m.bias(), 0)

    def test_psi_monitor(self):
        a = [0.1] * 100 + [0.2] * 100
        b = [0.8] * 100 + [0.9] * 100
        res = psi_monitor(a, b)
        self.assertIn("psi", res)
        self.assertIn("level", res)

    def test_auc_monitor(self):
        m = RollingAUCMonitor(window=200)
        for i in range(30):
            y = 1 if i % 2 == 0 else 0
            m.update(y, 0.8 if y == 1 else 0.2)
        # can be None only if sklearn is unavailable
        if m.value() is not None:
            self.assertGreaterEqual(m.value(), 0.5)

    def test_detect_drift_ignores_corrupted_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            f = root / "features" / "BTCUSDT"
            f.mkdir(parents=True)
            (f / "part-000.jsonl").write_text(
                '\n'.join(
                    ['{"ret_1": 0.01}', "{bad json}", '{"ret_1": 0.02}', '{"ret_1": 0.30}', '{"ret_1": 0.31}']
                ),
                encoding="utf-8",
            )
            res = detect_drift(root, "BTCUSDT", threshold=0.05)
            self.assertIn("drift", res)
            self.assertIn("delta_ret_mean", res)




    def test_detect_drift_streaming_reader_handles_large_input(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "features" / "BTCUSDT"
            p.mkdir(parents=True)
            rows = []
            for i in range(120):
                rows.append(json.dumps({"ret_1": 0.0001 if i < 60 else 0.02}))
            rows.insert(40, "{bad")
            (p / "part-000.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
            out = detect_drift(root, "BTCUSDT", threshold=0.005)
            self.assertTrue(bool(out["drift"]))

if __name__ == "__main__":
    unittest.main()
