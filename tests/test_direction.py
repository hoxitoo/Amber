"""Direction check (amber/backtest/direction.py) against markets with known answers.

Validated before it saw real data (CLAUDE.md rule 2), at the real comparison
count (60): a book that says nothing must read `no_direction`; a book whose
imbalance leads the price must be found; and a book whose information sits
only in the rows AFTER the minute closes must read `no_direction` too — a
feature that took one row too many would find it.
"""

import json
import math
import random
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

from amber.backtest import direction as D
from amber.exchange.orderbook import COLUMNS, format_row

DAY = 86_400_000
T0 = 1_762_000_000_000 - 1_762_000_000_000 % DAY


def _write_market(root: Path, *, mode: str, seed: int, symbols: int = 10, days: int = 4,
                  kappa: float = 0.0012, sigma: float = 0.004) -> None:
    """mode: 'null' (book says nothing about the future; flow follows the
    minute's own return), 'planted' (a persistent imbalance drives the drift
    after it), 'leak' (the next minute's imbalance drives the drift)."""
    rng = random.Random(seed)
    n = days * 1440
    raw, logs = root / "raw", root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ledger = []
    for s in range(symbols):
        sym = f"S{s:02d}USDT"
        imb = [0.0] * n
        for t in range(1, n):
            if mode == "planted":
                imb[t] = 0.95 * imb[t - 1] + rng.gauss(0, 0.3)
            elif mode == "skewed":
                imb[t] = rng.uniform(0.05, 0.9)  # always bid-heavy, unrelated to what follows
            else:
                imb[t] = rng.uniform(-1, 1)
            imb[t] = max(-0.95, min(0.95, imb[t]))
        price, candles, rets = 100.0, [], []
        for t in range(n):
            drift = 0.0006 if mode == "skewed" else 0.0
            if mode == "planted" and t >= 2:
                drift = kappa * imb[t - 2]
            elif mode == "leak" and t >= 1:
                drift = kappa * imb[t - 1]
            o, hi, lo = price, price, price
            for _ in range(8):
                price *= math.exp(drift / 8 + sigma / math.sqrt(8) * rng.gauss(0, 1))
                hi, lo = max(hi, price), min(lo, price)
            candles.append({"ts": T0 + t * D.MINUTE, "open": o, "high": hi, "low": lo, "close": price,
                            "is_synthetic": False})
            rets.append(price / o - 1)
        d = raw / "normalized" / sym
        d.mkdir(parents=True, exist_ok=True)
        (d / "part-000.jsonl").write_text("".join(json.dumps(c) + "\n" for c in candles))
        bdir = raw / "book" / sym
        bdir.mkdir(parents=True, exist_ok=True)
        lines = [",".join(COLUMNS)]
        for t in range(n):
            i = imb[t]
            if mode == "null":
                flow = max(-0.95, min(0.95, rets[t] / sigma * 0.5))  # flow follows its own minute
            elif mode == "skewed":
                flow = rng.uniform(0.0, 0.6)
            else:
                flow = rng.uniform(-0.5, 0.5)
            for k in range(1, 13):
                j = max(-0.99, min(0.99, i + rng.gauss(0, 0.05)))
                dep = 1e5
                feats = {"mid": candles[t]["close"], "spread_bps": 1.0,
                         "bid1_usd": 1e4 * (1 + j), "ask1_usd": 1e4 * (1 - j),
                         "bid_cov_bps": 50.0, "ask_cov_bps": 50.0}
                for b in (5, 10, 25, 50, 100):
                    feats[f"bid{b}_usd"] = dep * (1 + j)
                    feats[f"ask{b}_usd"] = dep * (1 - j)
                buy, sell = 1e3 * (1 + flow), 1e3 * (1 - flow)
                lines.append(format_row(T0 + t * D.MINUTE + k * D.ROW_MS, feats, [buy, sell, 3, 3, 500.0], 100))
        (bdir / "2026-11-01.csv").write_text("\n".join(lines) + "\n")
        labels = D.candle_labels(candles)
        for ts, lab in labels.items():
            if (ts // D.MINUTE) % 7 == 0:
                ledger.append({"status": "ok", "symbol": sym, "event_ts": ts, "first_touch": lab})
    (logs / "ignition_ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ledger))


def _run(mode, seed, **kw):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_market(root, mode=mode, seed=seed, **kw)
        syms = [f"S{s:02d}USDT" for s in range(kw.get("symbols", 10))]
        return D.run_direction_check(root / "raw", root / "logs", syms)


class TestLabels(unittest.TestCase):
    def _c(self, rows):
        return [{"ts": T0 + k * D.MINUTE, "open": c, "high": h, "low": lo, "close": c, "is_synthetic": s}
                for k, (h, lo, c, s) in enumerate(rows)]

    def test_entry_is_the_next_close_and_the_first_touch_wins(self):
        flat = (100, 100, 100, False)
        # bar 0 event, bar 1 entry close 100, bar 2 touches +1% only, bar 3 -1%
        rows = [flat, flat, (101.2, 99.9, 100, False), (100, 98.5, 100, False)] + [flat] * 40
        self.assertEqual(D.candle_labels(self._c(rows))[T0], 1)
        # a +1% wick on the entry bar itself is before entry and does not count
        rows = [flat, (102, 100, 100, False), (100.1, 98.9, 100, False)] + [flat] * 40
        self.assertEqual(D.candle_labels(self._c(rows))[T0], -1)

    def test_both_sides_in_one_bar_and_gaps_are_dropped(self):
        flat = (100, 100, 100, False)
        both = [flat, flat, (101.5, 98.5, 100, False)] + [flat] * 40
        self.assertNotIn(T0, D.candle_labels(self._c(both)))
        synth = [flat, flat, (100, 100, 100, True), (101.5, 100, 100, False)] + [flat] * 40
        self.assertNotIn(T0, D.candle_labels(self._c(synth)))


class TestFeatures(unittest.TestCase):
    def _book(self, rows):
        cols = ("ts", "mid", "bid1_usd", "ask1_usd", "bid5_usd", "ask5_usd", "bid10_usd", "ask10_usd",
                "buy_usd", "sell_usd")
        return {c: np.array([r[i] for r in rows], dtype=float) for i, c in enumerate(cols)}

    def test_a_minute_owns_rows_up_to_its_close_and_no_later(self):
        rows = []
        for k in range(1, 13):  # minute 0: bids heavy
            rows.append((T0 + k * D.ROW_MS, 100, 3, 1, 3, 1, 3, 1, 10, 0))
        for k in range(1, 13):  # minute 1: asks heavy
            rows.append((T0 + D.MINUTE + k * D.ROW_MS, 100, 1, 3, 1, 3, 1, 3, 0, 10))
        f = D.minute_features(self._book(rows))
        self.assertAlmostEqual(f[T0][0], 0.5)  # obi_l1 from the row at T0 + 60 s, minute 0's own
        self.assertAlmostEqual(f[T0 + D.MINUTE][0], -0.5)
        self.assertAlmostEqual(f[T0][4], 1.0)  # flow_1m
        self.assertAlmostEqual(f[T0 + D.MINUTE][5], 0.0)  # flow_5m: 120 buy vs 120 sell

    def test_a_minute_with_too_few_book_rows_is_not_scored(self):
        rows = [(T0 + k * D.ROW_MS, 100 if k > 4 else math.nan, 1, 1, 1, 1, 1, 1, 0, 0) for k in range(1, 13)]
        self.assertEqual(D.minute_features(self._book(rows)), {})


class TestKnownAnswers(unittest.TestCase):
    def test_a_book_that_says_nothing_reads_no_direction(self):
        verdicts = [_run("null", seed, symbols=6, days=3)["verdict"] for seed in range(3)]
        self.assertEqual(verdicts.count("direction_signal"), 0, verdicts)

    def test_a_leading_imbalance_is_found(self):
        r = _run("planted", 1)
        self.assertEqual(r["verdict"], "direction_signal", r["signals"])
        self.assertTrue(any("obi" in s or "model" in s for s in r["signals"]), r["signals"])

    def test_information_after_the_close_is_not_used(self):
        # The minute after the close moves ~0.5% the way its own book leans:
        # strong enough that any feature reading past the close finds it.
        r = _run("leak", 2, kappa=0.01)
        self.assertNotEqual(r["verdict"], "direction_signal", r["signals"])
        # The trap has teeth: features taken one minute late find the signal.
        real = D.minute_features
        with unittest.mock.patch.object(D, "minute_features",
                                        lambda book: {k - D.MINUTE: v for k, v in real(book).items()}):
            late = _run("leak", 2, kappa=0.01)
        self.assertEqual(late["verdict"], "direction_signal")

    def test_a_trending_market_with_a_skewed_book_is_not_a_signal(self):
        """Rising prices, a book always heavier on the bid: 'up' is right more
        often for every feature, and no feature is better than saying 'up'."""
        r = _run("skewed", 3, symbols=6, days=3)
        self.assertGreater(r["arms"]["all"]["up_share"], 0.6)
        self.assertNotEqual(r["verdict"], "direction_signal", r["signals"])

    def test_report_prints(self):
        text = D.format_report(_run("null", 9, symbols=3, days=2))
        self.assertIn("ИТОГ:", text)


if __name__ == "__main__":
    unittest.main()
