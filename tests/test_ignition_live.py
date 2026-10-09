"""Live ignition warnings (amber/signals/ignition_live.py).

What would make the channel lie: scoring live bars with features built
differently from the training rows, counting a candle twice, spamming one
coin, and a forward check that credits coin choice as timing.
"""

import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from amber.backtest.ignition import DEFAULT_ARMS, FEATURES, _load_symbol, collect_arms, lead_from_rows
from amber.signals import ignition_live as IL
from amber.storage.state_store import StateStore
from tests.test_ignition import _write_market

REPO = Path(__file__).resolve().parents[1]
DAY = 86_400_000
T0 = 1_700_000_000_000


def _market(td: str, **kw) -> Path:
    return _write_market(Path(td), seed=kw.pop("seed", 1), planted=kw.pop("planted", True),
                         symbols=3, n=kw.pop("n", 4000), btc_leads=True, **kw)


class TestTrainServeParity(unittest.TestCase):
    def test_live_vectors_equal_the_training_rows(self):
        """For calm bars in the training set, the live path — given only the
        rows up to that bar — must build the identical vector."""
        with tempfile.TemporaryDirectory() as td:
            root = _market(td)
            arms, _n, names, _sp = collect_arms(root, specs=[IL.SPEC], max_candles_per_symbol=0,
                                                min_warmup_bars=60)
            arm = arms[0]
            x = np.frombuffer(arm.x, dtype=np.float32).reshape(-1, len(FEATURES))
            rows = {n: _load_symbol(root / "features" / n, 10**9) for n in names}
            btc_rows = _load_symbol(root / "features" / "BTCUSDT", 10**9)
            rng = random.Random(0)
            picks = [k for k in range(len(arm.y)) if rng.random() < 0.02][:60]
            self.assertGreater(len(picks), 20)
            for k in picks:
                sym, ts = names[arm.sym[k]], arm.ts[k]
                i = next(j for j, r in enumerate(rows[sym]) if int(r["ts"]) == ts)
                tail = rows[sym][: i + 1][-IL.TAIL_ROWS:]
                btc_tail = [r for r in btc_rows if int(r["ts"]) <= ts][-IL.TAIL_ROWS:]
                live = IL.live_vector(tail, lead_from_rows(btc_tail), {}, window=IL.SPEC.window,
                                      calm_pct=IL.SPEC.calm_pct, min_warmup_bars=60)
                self.assertIsNotNone(live, f"{sym}@{ts} calm in training, not live")
                np.testing.assert_allclose(np.asarray(live, dtype=np.float32), x[k], rtol=1e-6, atol=1e-7,
                                           err_msg=f"{sym}@{ts}")


class TestTrainAndScore(unittest.TestCase):
    def test_artifact_has_per_coin_thresholds_and_scores_calm_coins_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = _market(td, n=12000)  # past the 7-day training minimum
            models = Path(td) / "models"
            res = IL.train_ignition(root, models, days=0, budget=0.01)
            self.assertEqual(res["status"], "ok", res)
            art = json.loads(IL.latest_artifact_path(models).read_text())
            self.assertTrue(art["thresholds"])
            self.assertEqual(art["features"], list(FEATURES))
            scorer = IL.IgnitionScorer.load_latest(models)
            recs = scorer.score_rows(IL.read_tails(root))
            # Through the scanner hook, as the live scan calls it.
            from amber.pipeline import scanner_app

            cfg = {"ignition": {"enabled": True}}
            scanner_app._scan_ignition(cfg, root, models, Path(td) / "logs", StateStore(Path(td) / "state"))
            if recs:
                self.assertTrue((Path(td) / "logs" / IL.RECORDS_FILE).exists())
            for r in recs:
                self.assertTrue(0.0 <= r["prob"] <= 1.0)
                self.assertEqual(r["alert"], int(r["threshold"] is not None and r["score"] >= r["threshold"]))
                self.assertEqual((r["horizon_min"], r["target_up_pct"]), (DEFAULT_ARMS[0].horizon, 0.01))

    def test_too_little_history_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as td:
            root = _market(td, n=400)
            self.assertNotEqual(IL.train_ignition(root, Path(td) / "m", days=0)["status"], "ok")
            self.assertIsNone(IL.latest_artifact_path(Path(td) / "m"))


class TestScannerHookIsIsolated(unittest.TestCase):
    def test_a_broken_artifact_never_breaks_the_scan(self):
        from amber.pipeline import scanner_app

        with tempfile.TemporaryDirectory() as td:
            models = Path(td) / "models"
            (models / IL.ARTIFACT_DIR).mkdir(parents=True)
            (models / IL.ARTIFACT_DIR / "ignition_20260101T000000Z_x.json").write_text("{broken")
            scanner_app._IGNITION_CACHE.update({"path": None, "scorer": None})
            n = scanner_app._scan_ignition({"ignition": {"enabled": True}}, Path(td), models, Path(td) / "logs",
                                           StateStore(Path(td) / "state"))
            self.assertEqual(n, 0)

    def test_disabled_does_nothing(self):
        from amber.pipeline import scanner_app

        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(scanner_app._scan_ignition({"ignition": {"enabled": False}}, Path(td), Path(td),
                                                        Path(td), StateStore(Path(td) / "s")), 0)


class _FakeScorer:
    def __init__(self, batches):
        self.batches = list(batches)

    def score_rows(self, _rows):
        return [dict(r) for r in self.batches.pop(0)]


def _rec(sym, ts, alert):
    return {"event_ts": ts, "symbol": sym, "score": 0.5, "prob": 0.1, "threshold": 0.4,
            "coin_base": 0.01, "alert": alert, "horizon_min": 30, "target_up_pct": 0.01}


class TestScan(unittest.TestCase):
    def test_one_record_per_candle_and_one_shown_per_coin_per_horizon(self):
        with tempfile.TemporaryDirectory() as td:
            logs, state = Path(td) / "logs", StateStore(Path(td) / "state")
            m = 60_000
            batches = [
                [_rec("A", T0, 1)],
                [_rec("A", T0, 1)],                 # same candle rescanned
                [_rec("A", T0 + 5 * m, 1)],         # still over threshold 5 min later
                [_rec("A", T0 + 31 * m, 1)],        # past the 30-min pause
            ]
            fake = _FakeScorer(batches)
            out = []
            with mock.patch.object(IL, "read_tails", return_value={}):
                for _ in batches:
                    out.append(IL.scan_ignition(Path(td), Path(td), logs, state, scorer=fake))
            self.assertEqual([len(o) for o in out], [1, 0, 1, 1])
            self.assertEqual([o[0]["notify"] for o in out if o], [1, 0, 1])
            lines = (logs / IL.RECORDS_FILE).read_text().splitlines()
            self.assertEqual(len(lines), 3)

    def test_text_says_calm_probability_base_and_no_direction(self):
        text = IL.alert_text({**_rec("ETHUSDT", T0, 1), "prob": 0.093, "coin_base": 0.012,
                              "factors": ["размах цены за 4 ч"]})
        for part in ("ETHUSDT", "спокойно", "9.3%", "обычно 1.2%", "Направление не прогнозируется", "размах цены за 4 ч"):
            self.assertIn(part, text)


def _ledger(logs: Path, rows):
    logs.mkdir(parents=True, exist_ok=True)
    (logs / IL.LEDGER_FILE).write_text("".join(json.dumps({"status": "ok", **r}) + "\n" for r in rows))


def _calm_ledger(*, days, coins, hit_alert, hit_calm, seed=0, alerts_on=None):
    """A calm bar per coin every 10 minutes; every 5th bar of each coin alerts."""
    rng = random.Random(seed)
    rows = []
    for k in range(days * 144):
        ts = T0 + k * 600_000
        for c, base in coins.items():
            alert = int(k % 5 == 0 and (alerts_on is None or c in alerts_on))
            p = hit_alert(base) if alert else hit_calm(base)
            rows.append({"symbol": c, "event_ts": ts, "alert": alert, "move_hit": int(rng.random() < p)})
    return rows


class TestForwardSummary(unittest.TestCase):
    COINS = {"CALMUSDT": 0.01, "JUMPYUSDT": 0.10}

    def test_alerts_no_better_than_calm_are_not_confirmed(self):
        with tempfile.TemporaryDirectory() as td:
            _ledger(Path(td), _calm_ledger(days=10, coins=self.COINS, hit_alert=lambda b: b, hit_calm=lambda b: b))
            self.assertIn(IL.summarize_ignition(Path(td))["verdict"], ("not_confirmed",))

    def test_alerts_on_the_jumpy_coin_alone_are_coin_choice_not_confirmation(self):
        """Alerts only on the coin that moves more often, but no better than
        its own calm bars: a pooled base rate would call this a 3x lift."""
        coins = {"CALMUSDT": 0.005, "JUMPYUSDT": 0.20}  # a pooled base would read ~2x
        with tempfile.TemporaryDirectory() as td:
            _ledger(Path(td), _calm_ledger(days=20, coins=coins, hit_alert=lambda b: b,
                                           hit_calm=lambda b: b, alerts_on={"JUMPYUSDT"}))
            s = IL.summarize_ignition(Path(td))
            self.assertLess(s["lift"], 1.4)
            self.assertEqual(s["verdict"], "not_confirmed")

    def test_a_real_timing_edge_is_confirmed(self):
        with tempfile.TemporaryDirectory() as td:
            _ledger(Path(td), _calm_ledger(days=10, coins=self.COINS, hit_alert=lambda b: min(1, 4 * b),
                                           hit_calm=lambda b: b))
            self.assertEqual(IL.summarize_ignition(Path(td))["verdict"], "confirmed")

    def test_under_seven_days_is_underpowered_however_strong(self):
        with tempfile.TemporaryDirectory() as td:
            _ledger(Path(td), _calm_ledger(days=3, coins=self.COINS, hit_alert=lambda b: min(1, 6 * b),
                                           hit_calm=lambda b: b))
            self.assertEqual(IL.summarize_ignition(Path(td))["verdict"], "underpowered")


class TestDashboardNeverReadsTheWholeLedger(unittest.TestCase):
    def test_view_reads_the_published_summary(self):
        from amber.dashboard import data as D

        with tempfile.TemporaryDirectory() as td:
            logs = Path(td)
            (logs / IL.RECORDS_FILE).write_text(json.dumps(_rec("A", T0, 1) | {"notify": 1}) + "\n")
            with mock.patch.object(IL, "summarize_ignition", side_effect=AssertionError("full ledger read")):
                view = D.ignition_view(logs)
            self.assertEqual(view["summary"], {"verdict": "pending"})
            IL.save_summary(logs, {"verdict": "underpowered", "days": 2})
            self.assertEqual(D.ignition_view(logs)["summary"]["verdict"], "underpowered")


class TestPipelineSchedule(unittest.TestCase):
    def _loop(self):
        spec = importlib.util.spec_from_file_location("run_pipeline_loop", REPO / "scripts" / "run_pipeline_loop.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_failed_training_is_not_retried_every_cycle(self):
        mod = self._loop()
        with tempfile.TemporaryDirectory() as td:
            cfg = {"ignition": {"enabled": True, "retrain_min": 1440},
                   "storage": {"models_dir": str(Path(td) / "models"), "features_dir": str(Path(td) / "f")}}
            with mock.patch("amber.signals.ignition_live.train_ignition",
                            return_value={"status": "not_enough_data"}) as train:
                self.assertTrue(mod._train_ignition_if_due(cfg, now=10_000.0))
                self.assertFalse(mod._train_ignition_if_due(cfg, now=10_090.0))
                self.assertTrue(mod._train_ignition_if_due(cfg, now=10_000.0 + mod.IGNITION_RETRY_SEC))
            self.assertEqual(train.call_count, 2)

    def test_disabled_never_trains(self):
        mod = self._loop()
        with mock.patch("amber.signals.ignition_live.train_ignition") as train:
            self.assertFalse(mod._train_ignition_if_due({"ignition": {"enabled": False}, "storage": {}}, now=1e9))
        train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
