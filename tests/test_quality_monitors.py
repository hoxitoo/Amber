import json
import unittest
import tempfile
from pathlib import Path

from amber.monitoring.drift import (
    PredictionBiasMonitor,
    RollingAUCMonitor,
    detect_drift,
    psi_from_quantile_reference,
    psi_monitor,
)


class TestQualityMonitors(unittest.TestCase):
    def test_bias_monitor(self):
        m = PredictionBiasMonitor(window=50)
        for _ in range(25):
            m.update(0.7, 0.3)
        self.assertIsNotNone(m.bias())
        self.assertGreater(m.bias(), 0)

    def test_psi_monitor_flags_disjoint_distributions_high(self):
        a = [0.1] * 100 + [0.2] * 100
        b = [0.8] * 100 + [0.9] * 100
        res = psi_monitor(a, b)
        self.assertGreater(res["psi"], 0.2)
        self.assertEqual(res["level"], "high")

    def test_psi_monitor_identical_distributions_low(self):
        a = [0.1, 0.2, 0.3, 0.4, 0.5] * 40
        res = psi_monitor(a, list(a))
        self.assertLess(res["psi"], 0.1)
        self.assertEqual(res["level"], "low")

    def test_psi_from_quantile_reference(self):
        # reference deciles of uniform [0,1); identical live data -> ~0 PSI
        edges = [i / 10 for i in range(11)]
        live_same = [i / 100 for i in range(100)]
        self.assertLess(psi_from_quantile_reference(edges, live_same), 0.05)
        # all live mass in the top bin -> large PSI
        live_shifted = [0.99] * 100
        self.assertGreater(psi_from_quantile_reference(edges, live_shifted), 1.0)

    def test_auc_monitor_perfectly_separable_is_high(self):
        m = RollingAUCMonitor(window=200)
        for i in range(30):
            y = 1 if i % 2 == 0 else 0
            m.update(y, 0.8 if y == 1 else 0.2)
        self.assertIsNotNone(m.value())
        self.assertGreater(m.value(), 0.99)

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
            res = detect_drift(root, "BTCUSDT")
            self.assertIn("drift", res)
            self.assertIn("delta_ret_mean", res)
            self.assertIn("max_psi", res)

    def test_detect_drift_flags_regime_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "features" / "BTCUSDT"
            p.mkdir(parents=True)
            rows = []
            for i in range(120):
                rows.append(json.dumps({"ret_1": 0.0001 if i < 60 else 0.02}))
            rows.insert(40, "{bad")
            (p / "part-000.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
            out = detect_drift(root, "BTCUSDT", threshold=0.2)
            self.assertTrue(bool(out["drift"]))
            self.assertEqual(out["level"], "high")

    def test_detect_drift_with_train_reference(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "features" / "BTCUSDT"
            p.mkdir(parents=True)
            # live ret_1 sits far outside the train reference grid
            rows = [json.dumps({"ret_1": 0.5}) for _ in range(50)]
            (p / "part-000.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
            reference = {"ret_1": [-0.01 + i * 0.002 for i in range(11)]}
            out = detect_drift(root, "BTCUSDT", reference=reference)
            self.assertTrue(out["drift"])
            self.assertEqual(out["reference"], "train_quantiles")
            self.assertIn("ret_1", out["per_feature"])


if __name__ == "__main__":
    unittest.main()
