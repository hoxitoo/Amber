import unittest

from amber.monitoring.drift import PredictionBiasMonitor, RollingAUCMonitor, psi_monitor


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


if __name__ == "__main__":
    unittest.main()
