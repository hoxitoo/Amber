import unittest

from amber.signals.scorer import calibrated_prob


class TestCalibration(unittest.TestCase):
    def test_isotonic_interpolation(self):
        calib = {"method": "isotonic", "x_thresholds": [0.0, 0.5, 1.0], "y_thresholds": [0.0, 0.6, 1.0]}
        p = calibrated_prob(0.25, calib)
        self.assertGreater(p, 0.25)
        self.assertLessEqual(p, 1.0)


if __name__ == "__main__":
    unittest.main()
