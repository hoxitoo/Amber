"""Clustered bounds: exact, never narrowed by clustering, honest at small counts.

Every bound in the project — label sweep, baseline check, operating curve,
forward ledger — rounded `rate * episodes` to whole hits first. Rounding up
made the "clustered" bound exceed the naive one, and on a null fixture a
random feature was declared a precursor (2026-10-04).
"""

import itertools
import unittest

from amber.backtest.label_sweep import _precision_at_budget, _wilson_low, clustered_low


class TestClusteredBound(unittest.TestCase):
    def test_never_above_the_bound_on_every_alert(self):
        for z in (1.96, 3.5):
            for alerts, eps in itertools.product(range(1, 41), range(1, 41)):
                if eps > alerts:
                    continue
                for hits in range(alerts + 1):
                    # Same exact method, every alert treated as independent.
                    naive = clustered_low(hits / alerts, alerts, z)
                    self.assertLessEqual(
                        clustered_low(hits / alerts, eps, z), naive + 1e-12,
                        f"hits={hits} alerts={alerts} eps={eps} z={z}",
                    )

    def test_the_incident(self):
        """2 hits in 15 alerts over 12 episodes, base rate 1.16%, z = 3.5."""
        base = 0.011620400258231117
        self.assertLess(clustered_low(2 / 15, 12, 3.5) / base, 1.0, "one lucky draw read as evidence")
        # The Wilson interval it replaced said lift 1.47 here on all 15 alerts.
        self.assertGreater(_wilson_low(2, 15, 3.5) / base, 1.0)

    def test_large_counts_agree_with_wilson(self):
        """Where the ledger lives: 87% over 131 episodes."""
        exact = clustered_low(0.87, 131, 2.39)
        self.assertAlmostEqual(exact, _wilson_low(114, 131, 2.39), delta=0.02)

    def test_the_shared_precision_helper_obeys_it(self):
        # 15 alerts at the top, 2 hits, spread over 12 episodes.
        scores = list(range(100, 0, -1))
        labels = [1 if i in (0, 5) else 0 for i in range(100)]
        ts = [i * 20 * 60_000 if i < 12 else (11 * 20 + i) * 60_000 for i in range(100)]
        res = _precision_at_budget(scores, labels, ts, budget=0.15, z=3.5, horizon=15)
        self.assertLessEqual(res["lift_ci_low_clustered"], res["lift_ci_low"] + 1e-12)


if __name__ == "__main__":
    unittest.main()
