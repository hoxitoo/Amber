"""Forward check of the probability a warning shows ("P = x%").

The owner asked to show "уверенность" on each warning. A probability is only
worth showing if, forward, about x% of warnings that say x% are followed by
the move; this pins the check that answers it.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from amber.backtest.label_sweep import clustered_high, clustered_low
from amber.signals import ignition_live as IL

T0 = 1_760_000_000_000 - (1_760_000_000_000 % 86_400_000)


def _rows(days, shown_prob, true_rate, seed=0, per_day=40):
    rng = random.Random(seed)
    rows = []
    for k in range(days * per_day):
        ts = T0 + k * (86_400_000 // per_day)  # 36 min apart: one episode each
        rows.append({"symbol": "AUSDT", "event_ts": ts, "alert": 1, "notify": 1, "prob": shown_prob,
                     "move_hit": int(rng.random() < true_rate)})
        rows.append({"symbol": "AUSDT", "event_ts": ts + 60_000, "alert": 0, "notify": 0, "prob": 0.005,
                     "move_hit": int(rng.random() < 0.005)})
    return rows


def _calibration(rows):
    with tempfile.TemporaryDirectory() as td:
        logs = Path(td)
        (logs / IL.LEDGER_FILE).write_text("".join(json.dumps({"status": "ok", **r}) + "\n" for r in rows))
        return IL.summarize_ignition(logs)["calibration"]


class TestShownProbability(unittest.TestCase):
    def test_an_honest_probability_reads_consistent(self):
        verdicts = [_calibration(_rows(10, 0.06, 0.06, seed=s))["verdict"] for s in range(10)]
        self.assertGreaterEqual(verdicts.count("consistent"), 9, verdicts)

    def test_a_message_that_over_promises_is_overstated(self):
        self.assertEqual(_calibration(_rows(10, 0.12, 0.04))["verdict"], "overstated")

    def test_a_message_that_under_promises_is_understated(self):
        self.assertEqual(_calibration(_rows(10, 0.02, 0.10))["verdict"], "understated")

    def test_under_seven_days_is_underpowered(self):
        self.assertEqual(_calibration(_rows(4, 0.12, 0.01))["verdict"], "underpowered")

    def test_bands_cover_every_scored_bar(self):
        rows = _rows(10, 0.06, 0.06)
        c = _calibration(rows)
        self.assertEqual(sum(b["bars"] for b in c["bands"]), len(rows))
        self.assertEqual(c["shown"], len(rows) // 2)

    def test_report_prints_the_verdict(self):
        text = IL.format_ignition_summary(IL.summarize_ignition(self._logs(_rows(10, 0.12, 0.04))))
        self.assertIn("ИТОГ вероятности: overstated", text)

    def _logs(self, rows):
        td = Path(tempfile.mkdtemp())
        (td / IL.LEDGER_FILE).write_text("".join(json.dumps({"status": "ok", **r}) + "\n" for r in rows))
        return td


class TestUpperBound(unittest.TestCase):
    def test_mirrors_the_lower_bound(self):
        self.assertAlmostEqual(clustered_high(0.5, 100, 1.96), 1 - clustered_low(0.5, 100, 1.96), places=9)
        self.assertGreater(clustered_high(0.06, 120, 1.96), 0.06)
        self.assertEqual(clustered_high(1.0, 50), 1.0)


if __name__ == "__main__":
    unittest.main()
