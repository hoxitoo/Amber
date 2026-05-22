import unittest

from amber.pipeline.scanner_app import _build_alert_limiter


class TestScannerAlerts(unittest.TestCase):
    def test_build_alert_limiter_uses_non_negative_cooldown(self):
        lim = _build_alert_limiter({"cooldown_sec": 15})
        self.assertEqual(lim.cooldown_sec, 15)

        lim2 = _build_alert_limiter({"cooldown_sec": -5})
        self.assertEqual(lim2.cooldown_sec, 0)


if __name__ == "__main__":
    unittest.main()
