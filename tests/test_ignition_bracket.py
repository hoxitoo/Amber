"""The breakout bracket booked on shown ignition warnings.

The bracket is the first rule here that tries to make money from a warning
that names no direction. Its conventions (what a 1m bar is assumed to have
done first) decide the sign of the answer, so they are pinned bar by bar, and
the whole rule is run on a market with no edge at all, where it must not look
profitable.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from amber.storage.state_store import StateStore
from amber.monitoring import ledger as L
from amber.signals import ignition_live as IL

T0 = 1_760_000_000_000 - (1_760_000_000_000 % 86_400_000)
M = 60_000
COST = 0.0009
D = IL.BRACKET_TRIGGER


def _bar(close, high=None, low=None):
    return {"close": close, "high": close if high is None else high, "low": close if low is None else low}


def _run(bars, target=0.01):
    """Reference bar (close 100) then `bars`; the window ends at the last one."""
    candles = [_bar(100.0), *bars]
    return L.bracket_trade(candles, 0, len(candles) - 1, trigger=D, target=target, cost=COST)


class TestBracketBarByBar(unittest.TestCase):
    UP, DN = 100 * (1 + D), 100 * (1 - D)

    def test_no_fill_is_no_trade(self):
        self.assertEqual(_run([_bar(100.1, 100.2, 99.8)] * 5), (None, None))

    def test_long_reaches_target(self):
        side, net = _run([_bar(100.35, 100.4, 99.95), _bar(100.8, 101.0, 100.3)])
        self.assertEqual(side, 1)
        self.assertAlmostEqual(net, 101.0 / self.UP - 1 - COST)

    def test_short_reaches_target(self):
        side, net = _run([_bar(99.65, 100.05, 99.6), _bar(99.2, 99.7, 99.0)])
        self.assertEqual(side, -1)
        self.assertAlmostEqual(net, 1 - 99.0 / self.DN - COST)

    def test_stop_is_the_other_order(self):
        side, net = _run([_bar(100.35, 100.4, 100.0), _bar(99.8, 100.3, 99.65)])
        self.assertEqual(side, 1)
        self.assertAlmostEqual(net, self.DN / self.UP - 1 - COST)

    def test_a_return_to_the_price_is_not_a_stop(self):
        side, net = _run([_bar(99.95, 100.4, 99.9), _bar(99.8, 99.9, 99.75)])
        self.assertEqual(side, 1)
        self.assertAlmostEqual(net, 99.8 / self.UP - 1 - COST)

    def test_later_bar_touching_stop_and_target_is_the_stop(self):
        side, net = _run([_bar(100.35, 100.4, 100.0), _bar(100.5, 101.2, 99.6)])
        self.assertEqual(side, 1)
        self.assertAlmostEqual(net, self.DN / self.UP - 1 - COST)

    def test_fill_bar_reaching_the_target_wins(self):
        side, net = _run([_bar(100.9, 101.1, 99.98)])
        self.assertEqual(side, 1)
        self.assertAlmostEqual(net, 101.0 / self.UP - 1 - COST)

    def test_both_orders_in_one_bar_is_the_worse_loss(self):
        side, net = _run([_bar(100.0, 100.5, 99.5)])
        self.assertEqual(side, 0)
        self.assertAlmostEqual(net, 1 - self.UP / self.DN - COST)

    def test_open_at_the_end_exits_on_the_close(self):
        side, net = _run([_bar(100.35, 100.4, 100.0), _bar(100.6, 100.7, 100.2)])
        self.assertEqual(side, 1)
        self.assertAlmostEqual(net, 100.6 / self.UP - 1 - COST)


def _walk(n, rng, vol=0.0012, start=100.0, sub=20):
    """1m bars from a 20-step random walk: real highs and lows, no edge.
    Returns the bars and the paths inside them."""
    bars, paths, p = [_bar(start)], [], start
    step = vol / sub ** 0.5
    for _ in range(n):
        path = [p]
        for _ in range(sub):
            p *= 1 + rng.gauss(0, step)
            path.append(p)
        paths.append(path)
        bars.append(_bar(p, max(path), min(path)))
    return bars, paths


def _path_truth(paths, ref=100.0, target=0.01):
    """The bracket traded on the path itself, every order at its own level."""
    up, dn = ref * (1 + D), ref * (1 - D)
    side = fill = stop = 0
    for path in paths:
        for x in path[1:]:
            if side == 0:
                if x >= up or x <= dn:
                    side, fill, stop = (1, up, dn) if x >= up else (-1, dn, up)
                continue
            if (side > 0 and x <= stop) or (side < 0 and x >= stop):
                return side * (stop / fill - 1)
            tp = ref * (1 + side * target)
            if (side > 0 and x >= tp) or (side < 0 and x <= tp):
                return side * (tp / fill - 1)
    return None if side == 0 else side * (paths[-1][-1] / fill - 1)


class TestNullMarket(unittest.TestCase):
    def test_bar_conventions_add_nothing_over_the_true_path(self):
        """What 1m bars hide must not turn into profit. The same brackets,
        traded on the sub-minute path and on its bars: the bars may only be
        worse. Calm and breaking-out volatility both, because a stop back at
        the price was exact when calm and +3.2 bps a trade at 0.5%/min."""
        for vol in (0.0012, 0.005):
            rng = random.Random(7)
            diffs = []
            for _ in range(3000):
                bars, paths = _walk(30, rng, vol=vol, sub=60)
                _, gross = L.bracket_trade(bars, 0, 30, trigger=D, target=0.01, cost=0.0)
                truth = _path_truth(paths)
                if gross is not None and truth is not None:
                    diffs.append(gross - truth)
            self.assertGreater(len(diffs), 1500)
            self.assertLessEqual(max(diffs), 1e-12, f"bars beat the path at vol {vol}")

    def test_a_random_walk_is_not_profitable_at_the_booked_cost(self):
        rng = random.Random(11)
        rows = []
        for k in range(10 * 40):
            bars, _ = _walk(30, rng)
            _, gross = L.bracket_trade(bars, 0, 30, trigger=D, target=0.01, cost=0.0)
            rows.append({"symbol": "AUSDT", "event_ts": T0 + k * 2_160_000, "alert": 1, "notify": 1,
                         "move_hit": 0, "bracket_net": None if gross is None else gross - L.BRACKET_COST})
        with tempfile.TemporaryDirectory() as td:
            _ledger(Path(td), rows)
            b = IL.summarize_ignition(Path(td))["bracket"]
        self.assertGreaterEqual(b["episodes"], 30)
        self.assertEqual(b["verdict"], "not_shown_profitable")


def _ledger(logs, rows):
    logs.mkdir(parents=True, exist_ok=True)
    (logs / IL.LEDGER_FILE).write_text("".join(json.dumps({"status": "ok", **r}) + "\n" for r in rows))


def _shown(days, net_fn, seed=0, per_day=40):
    rng = random.Random(seed)
    rows = []
    for k in range(days * per_day):
        ts = T0 + k * (86_400_000 // per_day)
        rows.append({"symbol": "AUSDT", "event_ts": ts, "alert": 1, "notify": 1, "move_hit": 0,
                     "bracket_side": 1, "bracket_net": net_fn(rng)})
        rows.append({"symbol": "AUSDT", "event_ts": ts + M, "alert": 0, "notify": 0, "move_hit": 0,
                     "bracket_side": None, "bracket_net": -0.5})  # not shown: never counted
    return rows


def _coin(rng, p_win):
    return (0.007 if rng.random() < p_win else -0.003) - COST


class TestBracketVerdict(unittest.TestCase):
    def _summary(self, rows):
        with tempfile.TemporaryDirectory() as td:
            _ledger(Path(td), rows)
            return IL.summarize_ignition(Path(td))["bracket"]

    def test_a_coin_flip_that_pays_the_fee_is_not_profitable(self):
        b = self._summary(_shown(10, lambda r: _coin(r, 0.30)))
        self.assertEqual(b["verdict"], "not_shown_profitable")

    def test_a_real_edge_is_profitable(self):
        b = self._summary(_shown(10, lambda r: _coin(r, 0.60)))
        self.assertEqual(b["verdict"], "profitable")
        self.assertEqual(b["shown_scored"], 400)  # the unshown bars are not in it

    def test_under_seven_days_is_underpowered_however_strong(self):
        self.assertEqual(self._summary(_shown(4, lambda r: _coin(r, 0.9)))["verdict"], "underpowered")

    def test_unfilled_brackets_are_not_trades(self):
        b = self._summary(_shown(10, lambda r: None if r.random() < 0.5 else _coin(r, 0.3)))
        self.assertLess(b["filled"], b["shown_scored"] * 0.6)
        self.assertGreater(b["filled"], b["shown_scored"] * 0.4)


def _write_candles(raw, symbol, closes, start_ts):
    d = raw / "normalized" / symbol
    d.mkdir(parents=True, exist_ok=True)
    with (d / "part-00000.jsonl").open("w") as fh:
        for k, c in enumerate(closes):
            fh.write(json.dumps({"ts": start_ts + k * M, **c}) + "\n")


class TestLedgerAndBackfill(unittest.TestCase):
    def _market(self, td):
        rng = random.Random(3)
        raw = Path(td) / "raw"
        _write_candles(raw, "AUSDT", _walk(600, rng)[0], T0)
        return raw

    def _record(self, k, notify=1):
        return {"event_ts": T0 + k * M, "symbol": "AUSDT", "score": 1.0, "prob": 0.1, "threshold": 0.5,
                "coin_base": 0.01, "alert": notify, "notify": notify, "horizon_min": 30,
                "target_up_pct": 0.01, "emitted_ms": T0 + (k + 1) * M + 5_000}

    def test_ignition_ledger_books_the_bracket_and_the_main_ledger_does_not(self):
        with tempfile.TemporaryDirectory() as td:
            raw, logs = self._market(td), Path(td) / "logs"
            logs.mkdir()
            recs = [self._record(k) for k in range(60, 400, 40)]
            state = StateStore(Path(td) / "state")
            state.set(IL.LEDGER_STATE_KEY, {"calm": 0})
            (logs / IL.RECORDS_FILE).write_text("".join(json.dumps(r) + "\n" for r in recs))
            L.update_ledger(logs, raw, state, sources={"calm": IL.RECORDS_FILE}, ledger_file=IL.LEDGER_FILE,
                            state_key=IL.LEDGER_STATE_KEY, bracket=IL.BRACKET_TRIGGER, now_ms=T0 + 600 * M)
            rows = [json.loads(x) for x in (logs / IL.LEDGER_FILE).read_text().splitlines()]
            self.assertEqual(len(rows), len(recs))
            self.assertTrue(all("bracket_net" in r for r in rows))
            filled = [r for r in rows if r["bracket_gross"] is not None]
            self.assertTrue(filled, "fixture never fills a bracket")
            for r in filled:  # booked at the bracket's own cost, not the main ledger's
                self.assertAlmostEqual(r["bracket_net"], r["bracket_gross"] - L.BRACKET_COST)
            # The main ledger is frozen (CLAUDE.md section 11 C): no new field.
            res = L.resolve_alert(*self._index(raw), T0 + 100 * M, horizon=15, barrier=0.01)
            self.assertNotIn("bracket_net", res)

    def _index(self, raw):
        from amber.monitoring.quality_report import _CandleIndex

        idx = _CandleIndex(raw)
        return idx.candles("AUSDT"), idx.timestamps("AUSDT")

    def test_backfill_scores_old_rows_once_with_the_same_entry(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "bf", Path(__file__).resolve().parents[1] / "scripts" / "backfill_ignition_bracket.py")
        bf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bf)
        with tempfile.TemporaryDirectory() as td:
            raw, logs = self._market(td), Path(td) / "logs"
            logs.mkdir()
            recs = [self._record(k, notify=int(k % 80 == 60)) for k in range(60, 400, 20)]
            state = StateStore(Path(td) / "state")
            state.set(IL.LEDGER_STATE_KEY, {"calm": 0})
            (logs / IL.RECORDS_FILE).write_text("".join(json.dumps(r) + "\n" for r in recs))
            # Resolved by the version before the bracket existed.
            L.update_ledger(logs, raw, state, sources={"calm": IL.RECORDS_FILE}, ledger_file=IL.LEDGER_FILE,
                            state_key=IL.LEDGER_STATE_KEY, now_ms=T0 + 600 * M)
            rows = [json.loads(x) for x in (logs / IL.LEDGER_FILE).read_text().splitlines()]
            shown = sum(r["notify"] for r in rows)
            self.assertEqual(IL.summarize_ignition(logs)["bracket"]["shown_scored"], 0)

            c = bf.backfill(logs, raw, now_ms=T0 + 600 * M)
            self.assertEqual((c["scored"], c["entry_mismatch"], c["no_candles"]), (shown, 0, 0))
            self.assertEqual(IL.summarize_ignition(logs)["bracket"]["shown_scored"], shown)
            again = bf.backfill(logs, raw, now_ms=T0 + 600 * M)
            self.assertEqual((again["scored"], again["already"]), (0, shown))

            # Same rule, same entry bar as a ledger that had the bracket from the start.
            fresh = Path(td) / "fresh"
            fresh.mkdir()
            state2 = StateStore(Path(td) / "state2")
            state2.set(IL.LEDGER_STATE_KEY, {"calm": 0})
            (fresh / IL.RECORDS_FILE).write_text((logs / IL.RECORDS_FILE).read_text())
            L.update_ledger(fresh, raw, state2, sources={"calm": IL.RECORDS_FILE}, ledger_file=IL.LEDGER_FILE,
                            state_key=IL.LEDGER_STATE_KEY, bracket=IL.BRACKET_TRIGGER, now_ms=T0 + 600 * M)
            self.assertEqual(IL.summarize_ignition(fresh)["bracket"], IL.summarize_ignition(logs)["bracket"])


class TestTelegram(unittest.TestCase):
    def test_text_gives_the_bracket_levels(self):
        text = IL.alert_text({"symbol": "ETHUSDT", "prob": 0.09, "coin_base": 0.012, "alert": 1,
                              "horizon_min": 30, "target_up_pct": 0.01, "price": 2500.0, "factors": []})
        for part in ("2507.50", "2492.50", "2525.00", "2475.00", "Направление не прогнозируется",
                     "ещё проверяется"):
            self.assertIn(part, text)

    def test_old_records_without_price_have_no_levels(self):
        text = IL.alert_text({"symbol": "ETHUSDT", "prob": 0.09, "coin_base": 0.012, "alert": 1,
                              "horizon_min": 30, "target_up_pct": 0.01})
        self.assertNotIn("брекет", text)

    def test_one_failed_send_stops_the_rest_of_the_scan(self):
        from amber.pipeline import scanner_app

        recs = [{"symbol": s, "notify": 1, "alert": 1, "prob": 0.1, "horizon_min": 30, "target_up_pct": 0.01}
                for s in ("A", "B", "C")]
        with tempfile.TemporaryDirectory() as td:
            models = Path(td)
            scanner_app._IGNITION_CACHE.update({"path": None, "scorer": None})
            with mock.patch.object(IL, "latest_artifact_path", return_value=models / "x.json"), \
                    mock.patch.object(IL.IgnitionScorer, "load_latest", return_value=object()), \
                    mock.patch.object(IL, "scan_ignition", return_value=recs), \
                    mock.patch("amber.alerts.telegram.send_telegram_text", return_value=False) as send:
                scanner_app._scan_ignition({"ignition": {"enabled": True, "telegram": True}}, models, models,
                                           models, StateStore(models / "s"))
            self.assertEqual(send.call_count, 1)

    def test_shipped_config_turns_telegram_on(self):
        from amber.common.config import ConfigLoader

        cfg = ConfigLoader(Path(__file__).resolve().parents[1]).load_yaml("config/amber.yaml")
        self.assertTrue(cfg["ignition"]["telegram"])


if __name__ == "__main__":
    unittest.main()
