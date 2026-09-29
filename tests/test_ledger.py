"""Forward ledger: scoring live alerts after the fact (CLAUDE.md section 11).

The ledger decides whether acting on alerts makes money, so the ways it can
lie are pinned here: booking a move that happened before a person could act,
reading a gap as "no move", scoring an alert twice across a restart, and
calling noise profitable.
"""

import json
import random
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from amber.monitoring.ledger import (
    MIN_EPISODES,
    format_summary,
    resolve_alert,
    summarize_ledger,
    update_ledger,
)
from amber.storage.state_store import StateStore

T0 = 1_700_000_000_000
STEP = 60_000
B = 0.01
COST = 0.0009


def _bars(closes, *, highs=None, lows=None, synthetic=()):
    out = []
    for i, c in enumerate(closes):
        prev = closes[i - 1] if i else c
        out.append({
            "ts": T0 + i * STEP,
            "open": prev,
            "high": highs[i] if highs else max(prev, c),
            "low": lows[i] if lows else min(prev, c),
            "close": c,
            "is_synthetic": i in synthetic,
        })
    return out


def _resolve(bars, alert_i, horizon=5, lag=1):
    return resolve_alert(
        bars, [b["ts"] for b in bars], T0 + alert_i * STEP,
        horizon=horizon, barrier=B, lag_bars=lag, cost=COST,
    )


class TestResolveAlert(unittest.TestCase):
    def test_momentum_wins_when_the_move_continues(self):
        # alert bar 1 closes up; entry on bar 2 at 101; bar 4 reaches +1%.
        bars = _bars([100, 101, 101, 101.5, 102.2, 102, 102, 102, 102])
        r = _resolve(bars, 1)
        self.assertEqual(r["status"], "ok")
        self.assertEqual((r["bar_dir"], r["first_touch"], r["move_hit"]), (1, 1, 1))
        self.assertAlmostEqual(r["momentum_net"], B - COST)
        self.assertAlmostEqual(r["fade_net"], -B - COST)

    def test_a_move_before_entry_is_not_booked(self):
        """The alert bar itself and the lag bar cannot be traded: a person
        sees the alert only after the bar closes."""
        # Huge move on the entry bar (index 2), flat afterwards.
        bars = _bars([100, 101, 105, 105, 105, 105, 105, 105, 105])
        r = _resolve(bars, 1)
        self.assertEqual(r["move_hit"], 0)
        self.assertAlmostEqual(r["momentum_net"], -COST)

    def test_both_barriers_in_one_bar_is_a_loss_for_either_rule(self):
        closes = [100, 101, 101, 101, 101, 101, 101, 101, 101]
        highs = [c * 1.001 for c in closes]
        lows = [c * 0.999 for c in closes]
        highs[3], lows[3] = 103, 99
        r = _resolve(_bars(closes, highs=highs, lows=lows), 1)
        self.assertEqual(r["first_touch"], 0)
        self.assertAlmostEqual(r["momentum_net"], -B - COST)
        self.assertAlmostEqual(r["fade_net"], -B - COST)

    def test_timeout_books_the_close(self):
        bars = _bars([100, 99, 99, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7])
        r = _resolve(bars, 1, horizon=5)
        self.assertIsNone(r["first_touch"])
        # down bar: momentum is short from 99 to the close of bar 1+1+5=7.
        self.assertAlmostEqual(r["momentum_net"], -(99.6 / 99 - 1) - COST)
        self.assertAlmostEqual(r["momentum_net"] + r["fade_net"], -2 * COST)

    def test_waits_until_the_horizon_has_elapsed(self):
        bars = _bars([100, 101, 101, 101.5, 101.7])
        self.assertIsNone(_resolve(bars, 1, horizon=5))

    def test_a_gap_filled_bar_is_not_read_as_no_move(self):
        bars = _bars([100, 101, 101, 101, 101, 101, 101, 101, 101], synthetic={4})
        self.assertEqual(_resolve(bars, 1)["status"], "gap")

    def test_a_missing_bar_is_a_gap(self):
        bars = _bars([100, 101, 101, 101, 101, 101, 101, 101, 101, 101])
        del bars[4]
        self.assertEqual(_resolve(bars, 1)["status"], "gap")

    def test_a_flat_alert_bar_has_no_trade(self):
        bars = _bars([100, 100, 101, 101, 101, 101, 101, 101, 101])
        r = _resolve(bars, 1)
        self.assertIsNone(r["momentum_net"])
        self.assertIsNone(r["fade_net"])


def _write_candles(raw: Path, symbol: str, bars) -> None:
    d = raw / "normalized" / symbol
    d.mkdir(parents=True, exist_ok=True)
    with (d / "part-000.jsonl").open("w", encoding="utf-8") as fh:
        for b in bars:
            fh.write(json.dumps({**b, "symbol": symbol}) + "\n")


def _model_signal(symbol: str, ts: int) -> str:
    iso = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
    return json.dumps({
        "event_ts": iso, "symbol": symbol, "horizon_min": 5, "target_up_pct": B,
        "prob_move_calibrated": 0.8, "model_version": "lightgbm_dual_v1",
        "market_context": {"model_run_id": "model_x"},
    })


class TestUpdateLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.logs, self.raw = self.tmp / "logs", self.tmp / "raw"
        self.logs.mkdir()
        self.state = StateStore(self.tmp / "state")
        self.closes = [100, 101, 101, 101.5, 102.2, 102, 102, 102, 102, 102]
        self._start()

    def _start(self):
        """Deploy: the ledger begins at whatever the logs hold now."""
        update_ledger(self.logs, self.raw, self.state, now_ms=T0)

    def test_history_from_before_the_deploy_is_not_scored(self):
        self.state = StateStore(self.tmp / "fresh_state")  # a box that never ran the ledger
        (self.logs / "signals.jsonl").write_text(_model_signal("AAAUSDT", T0 + STEP) + "\n")
        _write_candles(self.raw, "AAAUSDT", _bars(self.closes))
        self.assertEqual(update_ledger(self.logs, self.raw, self.state, now_ms=T0), 0)
        self.assertEqual(self._run(), 0, "scored an alert logged before the ledger existed")
        with (self.logs / "signals.jsonl").open("a") as fh:
            fh.write(_model_signal("AAAUSDT", T0 + STEP) + "\n")
        self.assertEqual(self._run(), 1)

    def _run(self, now_offset_bars=10):
        return update_ledger(self.logs, self.raw, self.state, lag_bars=1, cost=COST,
                             now_ms=T0 + now_offset_bars * STEP)

    def _ledger(self):
        p = self.logs / "ledger.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def test_scored_once_and_only_when_complete(self):
        (self.logs / "signals.jsonl").write_text(_model_signal("AAAUSDT", T0 + STEP) + "\n")
        _write_candles(self.raw, "AAAUSDT", _bars(self.closes[:5]))
        self.assertEqual(self._run(), 0, "scored before the horizon elapsed")

        _write_candles(self.raw, "AAAUSDT", _bars(self.closes))
        self.assertEqual(self._run(), 1)
        self.assertEqual(self._run(), 0, "scored the same alert twice")
        # A restart re-reads the offset from disk.
        self.state = StateStore(self.tmp / "state")
        self.assertEqual(self._run(), 0, "scored again after a restart")
        rec = self._ledger()[0]
        self.assertEqual((rec["source"], rec["status"], rec["model_run_id"]), ("model", "ok", "model_x"))
        self.assertAlmostEqual(rec["momentum_net"], B - COST)

    def test_a_half_written_line_is_left_for_the_next_cycle(self):
        _write_candles(self.raw, "AAAUSDT", _bars(self.closes))
        line = _model_signal("AAAUSDT", T0 + STEP)
        (self.logs / "signals.jsonl").write_text(line[:20])
        self.assertEqual(self._run(), 0)
        (self.logs / "signals.jsonl").write_text(line + "\n")
        self.assertEqual(self._run(), 1)

    def test_candles_that_never_arrive_expire_instead_of_blocking(self):
        (self.logs / "signals.jsonl").write_text(
            _model_signal("DEADUSDT", T0 + STEP) + "\n" + _model_signal("AAAUSDT", T0 + STEP) + "\n"
        )
        _write_candles(self.raw, "AAAUSDT", _bars(self.closes))
        _write_candles(self.raw, "DEADUSDT", _bars(self.closes[:3]))
        self.assertEqual(self._run(), 0, "the stalled symbol should hold the queue for now")
        self.assertEqual(self._run(now_offset_bars=7 * 60), 2)
        self.assertEqual([r["status"] for r in self._ledger()], ["no_data", "ok"])

    def test_a_backlog_older_than_the_candle_cap_is_written_off(self):
        from amber.monitoring.ledger import MAX_CANDLE_ROWS

        (self.logs / "signals.jsonl").write_text(_model_signal("AAAUSDT", T0 + STEP) + "\n")
        _write_candles(self.raw, "AAAUSDT", _bars(self.closes))
        self.assertEqual(self._run(now_offset_bars=MAX_CANDLE_ROWS + 10), 1)
        self.assertEqual(self._ledger()[0]["status"], "no_data")

    def test_shadow_alerts_are_scored_as_the_rule(self):
        _write_candles(self.raw, "AAAUSDT", _bars(self.closes))
        (self.logs / "shadow_signals.jsonl").write_text(json.dumps({
            "event_ts": T0 + STEP, "symbol": "AAAUSDT", "horizon_min": 5, "target_up_pct": B,
            "range_atr_14": 4.2, "threshold": 3.9, "model_run_id": "model_x",
        }) + "\n")
        self.assertEqual(self._run(), 1)
        rec = self._ledger()[0]
        self.assertEqual((rec["source"], rec["score"]), ("rule", 4.2))


def _ledger_rows(n_episodes, *, gap_min, per_episode=1, pnl):
    rows = []
    for e in range(n_episodes):
        for k in range(per_episode):
            m, f = pnl(e, k)
            rows.append({
                "source": "model", "status": "ok", "symbol": f"S{k}", "horizon": 15,
                "event_ts": T0 + e * gap_min * STEP, "move_hit": 1,
                "momentum_net": m, "fade_net": f,
            })
    return rows


def _write_ledger(logs: Path, rows) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


class TestSummary(unittest.TestCase):
    def test_noise_is_not_called_profitable(self):
        """A fair coin at ±1% minus costs, over many episodes: no rule may be
        declared profitable at the family-wise bound."""
        rng = random.Random(7)

        def coin(e, k):
            g = B if rng.random() < 0.5 else -B
            return g - COST, -g - COST

        logs = Path(tempfile.mkdtemp())
        for seed in range(20):
            rng.seed(seed)
            _write_ledger(logs, _ledger_rows(200, gap_min=30, pnl=coin))
            s = summarize_ledger(logs)["sources"]["model"]
            for rule in ("momentum", "fade"):
                self.assertNotEqual(s["rules"][rule]["verdict"], "profitable", f"seed {seed} {rule}")

    def test_a_real_edge_is_found(self):
        rng = random.Random(1)

        def edge(e, k):
            g = B if rng.random() < 0.75 else -B
            return g - COST, -g - COST

        logs = Path(tempfile.mkdtemp())
        _write_ledger(logs, _ledger_rows(120, gap_min=30, pnl=edge))
        s = summarize_ledger(logs)["sources"]["model"]
        self.assertEqual(s["rules"]["momentum"]["verdict"], "profitable")
        self.assertNotEqual(s["rules"]["fade"]["verdict"], "profitable")

    def test_simultaneous_alerts_are_one_episode(self):
        """27 symbols firing in one market lurch are one observation, not 27."""
        logs = Path(tempfile.mkdtemp())
        _write_ledger(logs, _ledger_rows(
            MIN_EPISODES - 1, gap_min=30, per_episode=27, pnl=lambda e, k: (B - COST, -B - COST),
        ))
        s = summarize_ledger(logs)["sources"]["model"]
        self.assertEqual(s["rules"]["momentum"]["trades"], 27 * (MIN_EPISODES - 1))
        self.assertEqual(s["rules"]["momentum"]["episodes"], MIN_EPISODES - 1)
        self.assertEqual(s["rules"]["momentum"]["verdict"], "underpowered")

    def test_format_runs_on_an_empty_ledger(self):
        text = format_summary(summarize_ledger(Path(tempfile.mkdtemp())))
        self.assertIn("underpowered", text)


if __name__ == "__main__":
    unittest.main()
