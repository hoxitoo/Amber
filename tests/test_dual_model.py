import unittest

from amber.models.infer import infer_raw_prob


class TestDualModelInfer(unittest.TestCase):
    def test_dual_heads_are_independent(self):
        model = {
            "heads": {
                "pump": {"weights": {"ret_1": 5.0, "vol_z_20": 0.0}, "bias": 0.0},
                "dump": {"weights": {"ret_1": -5.0, "vol_z_20": 0.0}, "bias": 0.0},
            }
        }
        up = infer_raw_prob(model, ret_1=0.02, vol_z_20=0.0, target="pump")
        down = infer_raw_prob(model, ret_1=0.02, vol_z_20=0.0, target="dump")
        self.assertGreater(up, down)


if __name__ == "__main__":
    unittest.main()
