"""The dashboard load path must stay bounded, and must not change its answers.

Measured on a realistic fixture (27 symbols x 62k candles, 20k signals) the
load took 151 seconds, which on the box itself was minutes of blank browser.
146 of those were one line: `_confirmed_outcome` rebuilt the candle timestamp
array on every signal — the same list every time, 1.2 billion element
constructions per report.

Two kinds of test here, and the distinction matters:

- Caching the timestamp array changes *nothing* about the result, so the
  equivalence tests below pin it exactly.
- Bounding the reads deliberately narrows what the monitors see. Both monitors
  are 200-wide, so nothing that affects their output is discarded, but the
  reported counts now describe the window and that has to be explicit.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from amber.dashboard.data import candle_stats
from amber.monitoring.quality_report import _CandleIndex, _confirmed_outcome, build_quality_report


def _write_candles(raw_root: Path, symbol: str, n: int, *, synthetic_every: int = 0) -> None:
    d = raw_root / "normalized" / symbol
    d.mkdir(parents=True, exist_ok=True)
    rng = random.Random(hash(symbol) % 10_000)
    price, ts = 100.0, 1_700_000_000_000
    with (d / "part-000.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(n):
            price *= 1 + rng.gauss(0, 0.002)
            fh.write(json.dumps({
                "ts": ts + i * 60_000, "symbol": symbol,
                "open": price, "high": price * 1.004, "low": price * 0.996,
                "close": price, "volume": 5.0, "mid_price": price,
                "is_synthetic": bool(synthetic_every and i % synthetic_every == 0),
            }) + "\n")


def _write_signals(logs: Path, n: int, symbols: list[str]) -> Path:
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / "signals.jsonl"
    ts0 = 1_700_000_000_000
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({
                "signal_id": f"sig_{i}", "symbol": symbols[i % len(symbols)],
                "event_ts": ts0 + i * 60_000 * 3, "horizon_min": 15,
                "target_up_pct": 0.01,
                "prob_up_calibrated": 0.3, "prob_down_calibrated": 0.2,
                "prob_move_calibrated": 0.5,
            }) + "\n")
    return path


class TestOutcomeConfirmationUnchanged(unittest.TestCase):
    """Caching the timestamp array is a pure speedup — the verdicts must match."""

    def test_every_signal_resolves_the_same_way_as_a_naive_scan(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw"
            _write_candles(raw, "AAAUSDT", 800)
            index = _CandleIndex(raw)
            candles = index.candles("AAAUSDT")

            def naive(event_ts: int, horizon: int, target: float) -> int | None:
                """The original formulation, written out independently."""
                ts_list = [int(c.get("ts", 0) or 0) for c in candles]
                from bisect import bisect_right

                entry_i = bisect_right(ts_list, event_ts) - 1
                if entry_i < 0:
                    return None
                entry = float(candles[entry_i]["close"])
                if entry <= 0 or target <= 0:
                    return None
                deadline = event_ts + horizon * 60_000
                if ts_list[-1] < deadline:
                    return None
                level = entry * (1.0 + target)
                for c in candles[entry_i + 1:]:
                    if int(c["ts"]) > deadline:
                        break
                    if float(c["high"]) >= level:
                        return 1
                return 0

            base_ts = 1_700_000_000_000
            checked = 0
            for step in range(0, 800, 7):
                event_ts = base_ts + step * 60_000
                self.assertEqual(
                    _confirmed_outcome(index, "AAAUSDT", event_ts, 15, 0.01),
                    naive(event_ts, 15, 0.01),
                    f"verdict changed at step {step}",
                )
                checked += 1
            self.assertGreater(checked, 50)

    def test_timestamps_match_the_candles_they_index(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw"
            _write_candles(raw, "AAAUSDT", 300)
            index = _CandleIndex(raw)
            self.assertEqual(
                index.timestamps("AAAUSDT"),
                [int(c["ts"]) for c in index.candles("AAAUSDT")],
            )


class TestQualityReportEquivalence(unittest.TestCase):
    """Within the window, the report must be what it always was."""

    def test_small_history_is_unaffected_by_the_bounds(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            symbols = ["AAAUSDT", "BBBUSDT"]
            for s in symbols:
                _write_candles(root / "raw", s, 900)
            path = _write_signals(root / "logs", 120, symbols)

            generous = build_quality_report(path, raw_root=root / "raw",
                                            max_signals=10**6, candle_tail=10**6)
            bounded = build_quality_report(path, raw_root=root / "raw")

            for key in ("rolling_auc", "prediction_bias", "auc_confirmed_outcomes",
                        "auc_unconfirmed_outcomes", "signals"):
                self.assertEqual(generous[key], bounded[key], f"{key} changed under bounding")

    def test_signal_total_is_exact_even_outside_the_window(self):
        """The KPI counts every signal ever emitted; only the outcome counts
        describe the window."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_candles(root / "raw", "AAAUSDT", 500)
            path = _write_signals(root / "logs", 5000, ["AAAUSDT"])

            rep = build_quality_report(path, raw_root=root / "raw", max_signals=100)

            self.assertEqual(rep["signals"], 5000)
            self.assertEqual(rep["signals_in_window"], 100)
            self.assertEqual(rep["window_max_signals"], 100)
            self.assertLessEqual(rep["auc_confirmed_outcomes"] + rep["auc_unconfirmed_outcomes"], 100)

    def test_reading_is_bounded_by_the_tail(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw"
            _write_candles(raw, "AAAUSDT", 5000)
            self.assertEqual(len(_CandleIndex(raw, max_rows=400).candles("AAAUSDT")), 400)
            self.assertEqual(len(_CandleIndex(raw).candles("AAAUSDT")), 5000)

    def test_the_tail_kept_is_the_newest_part(self):
        """Confirmation only looks forward, so the newest rows are the ones
        that matter; keeping the oldest would resolve nothing."""
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw"
            _write_candles(raw, "AAAUSDT", 1000)
            full = _CandleIndex(raw).candles("AAAUSDT")
            tail = _CandleIndex(raw, max_rows=50).candles("AAAUSDT")
            self.assertEqual([c["ts"] for c in tail], [c["ts"] for c in full[-50:]])


class TestCandleStats(unittest.TestCase):
    def test_the_count_stays_exact(self):
        """Counting newlines instead of parsing must not change the number."""
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw"
            _write_candles(raw, "AAAUSDT", 3333)
            stats = candle_stats(str(raw), ["AAAUSDT"])
            self.assertEqual(stats[0]["candles"], 3333)

    def test_synthetic_share_is_measured_over_the_recent_sample(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw"
            _write_candles(raw, "AAAUSDT", 1000, synthetic_every=4)
            stats = candle_stats(str(raw), ["AAAUSDT"], sample=200)

            self.assertEqual(stats[0]["synthetic_sample"], 200)
            self.assertAlmostEqual(stats[0]["synthetic_pct"], 25.0, delta=1.0)

    def test_last_update_comes_from_the_newest_row(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw"
            _write_candles(raw, "AAAUSDT", 500)
            full = candle_stats(str(raw), ["AAAUSDT"], sample=10**6)
            tail = candle_stats(str(raw), ["AAAUSDT"], sample=10)
            self.assertAlmostEqual(full[0]["last_update_min"], tail[0]["last_update_min"], places=3)

    def test_missing_symbol_does_not_raise(self):
        with tempfile.TemporaryDirectory() as td:
            stats = candle_stats(str(Path(td) / "raw"), ["NOPEUSDT"])
            self.assertEqual(stats[0]["candles"], 0)
            self.assertEqual(stats[0]["synthetic_pct"], 0.0)
            self.assertIsNone(stats[0]["last_update_min"])


if __name__ == "__main__":
    unittest.main()
