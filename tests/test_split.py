import unittest

from amber.models.split import make_walk_forward_folds


class TestSplit(unittest.TestCase):
    def test_folds_have_gap_and_order(self):
        folds = make_walk_forward_folds(n_rows=500, n_folds=3, val_size=50, gap=30)
        self.assertGreaterEqual(len(folds), 1)
        for f in folds:
            self.assertLessEqual(f.train_end + 30, f.val_start)
            self.assertLess(f.val_start, f.val_end)


if __name__ == "__main__":
    unittest.main()
