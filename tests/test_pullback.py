"""Pullback study (amber/backtest/pullback.py): conventions and null fixtures.

Validated before it saw real data (CLAUDE.md rule 2), at the real arm count
(12) and the real length (~6.7 years of 4H bars, two correlated symbols):
- no edge, no costs: no variant may read profitable more than by chance;
- a pure trend, no costs: variants may profit, but must not beat the
  random-entry control, because a dip in a random walk tells nothing;
- a planted bounce after a 3% drop: the drop variants must find it.
"""

import math
import random
import unittest
from unittest import mock

from amber.backtest import pullback as P

T0 = 1_577_836_800_000  # 2020-01-01
BARS = int(6.7 * 365 * 6)


def _market(seed, *, drift=0.0, vol=0.012, bounce=0.0, rho=0.8, n=BARS, sub=8):
    """Two correlated symbols of 4H bars with sub-bar paths (real highs/lows).
    `bounce`: extra drift per bar for 6 bars after a close 3% under the
    6-bar high — the planted support."""
    rng = random.Random(seed)
    step = vol / math.sqrt(sub)
    out = {}
    paths = {s: {"p": 100.0, "bars": [], "left": 0} for s in ("AUSDT", "BUSDT")}
    for k in range(n):
        common = [rng.gauss(0, 1) for _ in range(sub)]
        for s, st in paths.items():
            mu = drift + (bounce if st["left"] > 0 else 0.0)
            p0 = p = st["p"]
            hi = lo = p
            for x in common:
                e = rho * x + math.sqrt(1 - rho * rho) * rng.gauss(0, 1)
                p *= math.exp(mu / sub + step * e)
                hi, lo = max(hi, p), min(lo, p)
            st["bars"].append({"ts": T0 + k * P.TF_MS, "open": p0, "high": hi, "low": lo, "close": p})
            st["p"] = p
            st["left"] = max(0, st["left"] - 1)
            if bounce and len(st["bars"]) > 6:
                ref = max(b["high"] for b in st["bars"][-7:-1])
                if p <= ref * (1 - P.DROP_PCT):
                    st["left"] = 6
    for s, st in paths.items():
        out[s] = (st["bars"], [])
    return out


def _free(fn):
    """Run with zero fees and slippage: the null must hold before costs."""
    with mock.patch.object(P, "TAKER", 0.0), mock.patch.object(P, "MAKER", 0.0), mock.patch.object(P, "SLIP", 0.0):
        return fn()


def _bar(o, h, low, c, k):
    return {"ts": T0 + k * P.TF_MS, "open": o, "high": h, "low": low, "close": c}


class TestExitConventions(unittest.TestCase):
    def _series(self, rows):
        return P._Series([_bar(*r, k) for k, r in enumerate(rows)])

    def test_bar_touching_stop_and_target_is_a_stop(self):
        s = self._series([(100, 100, 100, 100), (100, 140, 80, 100)] + [(100, 100, 100, 100)] * 40)
        self.assertEqual(s.exit(0, 100, 90, 130, False)[2], "stop")

    def test_limit_fill_bar_can_stop_but_not_take_profit(self):
        s = self._series([(100, 140, 99, 100)] + [(100, 100, 100, 100)] * 40)
        self.assertEqual(s.exit(0, 100, 90, 130, True)[2], "timeout")
        s = self._series([(100, 101, 85, 100)] + [(100, 100, 100, 100)] * 40)
        self.assertEqual(s.exit(0, 100, 90, 130, True)[2], "stop")

    def test_gap_through_the_stop_fills_at_the_open(self):
        s = self._series([(100, 100, 100, 100), (80, 85, 79, 82)] + [(82, 82, 82, 82)] * 40)
        _, px, reason = s.exit(0, 100, 90, 130, False)
        self.assertEqual(reason, "stop")
        self.assertAlmostEqual(px, 80 * (1 - P.SLIP))

    def test_window_past_the_data_is_not_a_trade(self):
        s = self._series([(100, 100, 100, 100)] * 10)
        self.assertIsNone(s.exit(0, 100, 90, 130, False))


class TestMirrorAndFunding(unittest.TestCase):
    def test_short_is_the_long_rule_on_the_inverted_series(self):
        bars, f = _market(3)["AUSDT"]
        short = P.arm_trades(bars, f, "drop", "confirm", -1)
        long_inv = P.arm_trades(P.invert(bars), f, "drop", "confirm", 1)
        self.assertTrue(short)
        self.assertEqual([t["entry_ts"] for t in short], [t["entry_ts"] for t in long_inv])
        for s, lg in zip(short, long_inv):
            self.assertAlmostEqual(s["gross"], 1 - 1 / (1 + lg["gross"]), places=12)

    def test_longs_pay_positive_funding_and_shorts_receive_it(self):
        bars, _ = _market(4)["AUSDT"]
        funding = [(b["ts"], 0.001) for b in bars]  # 0.1% every bar
        for side in (1, -1):
            free = P.arm_trades(bars, [], "drop", "limit", side)
            paid = P.arm_trades(bars, funding, "drop", "limit", side)
            self.assertTrue(free)
            for a, b in zip(free, paid):
                self.assertEqual(-side * (b["net"] - a["net"]) > 0, True)


class TestNoLookAhead(unittest.TestCase):
    """Every entry must be decided by its own bar and the ones before it."""

    def _entry_bar(self, bars, tr):
        ts = tr["entry_ts"] - (0 if tr["arm"].split("/")[1] == "limit" else P.TF_MS)
        return next(k for k, b in enumerate(bars) if b["ts"] == ts)

    def test_entries_survive_a_flat_future(self):
        bars, f = _market(5)["AUSDT"]
        rng = random.Random(0)
        for branch in P.BRANCHES:
            for entry in P.ENTRIES:
                for side in (1, -1):
                    full = P.arm_trades(bars, f, branch, entry, side)
                    self.assertTrue(full, f"{branch}/{entry}/{side}: no trades")
                    for tr in rng.sample(full, min(15, len(full))):
                        e = self._entry_bar(bars, tr)
                        last = bars[e]
                        flat = [{"ts": last["ts"] + (k + 1) * P.TF_MS, "open": last["close"], "high": last["close"],
                                 "low": last["close"], "close": last["close"]} for k in range(P.TIMEOUT + 2)]
                        cut = P.arm_trades(bars[:e + 1] + flat, f, branch, entry, side)
                        self.assertIn(tr["entry_ts"], [t["entry_ts"] for t in cut], f"{branch}/{entry}/{side}")

    def test_a_limit_fill_does_not_know_its_own_close(self):
        bars, f = _market(6)["AUSDT"]
        for branch in P.BRANCHES:
            for side in (1, -1):
                full = P.arm_trades(bars, f, branch, "limit", side)
                for tr in full[:15]:
                    e = self._entry_bar(bars, tr)
                    b = bars[e]
                    # What the fill cannot know yet: where the bar closes, and
                    # how high it goes (for a long; how low, for a short).
                    for change in ({"close": b["low"]}, {"close": b["high"]},
                                   {"high": max(b["open"], b["close"])}, {"low": min(b["open"], b["close"])}):
                        if side > 0 and "low" in change or side < 0 and "high" in change:
                            continue  # the side the order fills on
                        alt = [dict(x) for x in bars]
                        alt[e].update(change)
                        alt_tr = P.arm_trades(alt[:e + P.TIMEOUT + 2], f, branch, "limit", side)
                        self.assertIn(tr["entry_ts"], [t["entry_ts"] for t in alt_tr], f"{branch}/{side}/{change}")


class TestNullFixtures(unittest.TestCase):
    def test_no_edge_no_costs_reads_profitable_only_by_chance(self):
        hits = []
        for seed in range(6):
            r = _free(lambda: P.run_study(_market(100 + seed)))
            hits.append([a["arm"] for a in r["arms"] if a["verdict"] in ("beats_trend", "matched_by_trend")])
            self.assertGreaterEqual(min(a["episodes"] for a in r["arms"]), P.MIN_EPISODES, "fixture underpowered")
        self.assertLessEqual(sum(1 for h in hits if h), 1, hits)

    def test_a_pure_trend_is_matched_by_the_control(self):
        beats = []
        for seed in range(6):
            r = _free(lambda: P.run_study(_market(200 + seed, drift=0.0006)))
            beats.append([a["arm"] for a in r["arms"] if a["verdict"] == "beats_trend"])
        self.assertLessEqual(sum(1 for b in beats if b), 1, beats)

    def test_a_planted_bounce_is_found(self):
        r = P.run_study(_market(300, drift=0.0003, bounce=0.006))
        by = {a["arm"]: a for a in r["arms"]}
        self.assertEqual(by["drop/limit/long"]["verdict"], "beats_trend", by["drop/limit/long"])


if __name__ == "__main__":
    unittest.main()
