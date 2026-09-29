"""The shadow range_atr_14 rule the forward ledger compares the model to.

It is only a fair comparison if the rule runs under the model's conditions —
same rate, same filters, same gate limits — without borrowing the model's
timing or taking its slots, and without ever breaking the real scan.
"""

import json
import tempfile
import unittest
from pathlib import Path

from amber.pipeline.scanner_app import _scan_shadow
from amber.signals.filters import SignalGate
from amber.signals.shadow import (
    GATE_STATE_KEY,
    RULE_FEATURE,
    SHADOW_SIGNALS_FILE,
    fit_shadow_rule,
    load_shadow_rule,
    save_shadow_rule,
    shadow_candidates,
)
from amber.storage.state_store import StateStore

MODEL = {"labeling": {"horizon_steps": 15, "avg_up_pct": 0.01}}


def _row(symbol, atr, **kw):
    return {"symbol": symbol, "ts": 1_700_000_000_000, "obs": 500, "is_synthetic": False,
            "bid": 99.99, "ask": 100.01, "mid_price": 100.0, RULE_FEATURE: atr, **kw}


class TestFit(unittest.TestCase):
    def test_cut_fires_on_the_models_share_of_rows(self):
        rows = [{RULE_FEATURE: float(v)} for v in range(1000)]
        rule = fit_shadow_rule(rows, 0.01, model_run_id="model_x")
        fired = sum(1 for r in rows if r[RULE_FEATURE] >= rule["threshold"])
        self.assertEqual(fired, 10)
        self.assertEqual(rule["model_run_id"], "model_x")

    def test_a_silent_model_gives_a_silent_rule(self):
        rule = fit_shadow_rule([{RULE_FEATURE: 1.0}] * 100, 0.0, model_run_id=None)
        self.assertIsNone(rule["threshold"])
        with tempfile.TemporaryDirectory() as td:
            save_shadow_rule(Path(td), rule)
            self.assertIsNone(load_shadow_rule(Path(td)))


class TestCandidates(unittest.TestCase):
    def test_same_filters_as_the_scanner_and_strongest_first(self):
        rule = {"feature": RULE_FEATURE, "threshold": 3.0}
        rows = [
            _row("LOW", 2.9),
            _row("MID", 3.5),
            _row("TOP", 5.0),
            _row("SYN", 9.0, is_synthetic=True),
            _row("NEW", 9.0, obs=10),
            _row("WIDE", 9.0, bid=99.0, ask=101.0),  # 200 bps
        ]
        got = [r["symbol"] for r in shadow_candidates(rows, rule, min_warmup=60, spread_max_bps=30.0)]
        self.assertEqual(got, ["TOP", "MID"])


class TestScanShadow(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.logs = self.tmp / "logs"
        self.store = StateStore(self.tmp / "state")
        save_shadow_rule(self.logs, {"feature": RULE_FEATURE, "threshold": 3.0, "model_run_id": "model_x"})

    def _gate(self, limit=5):
        return SignalGate(cooldown_sec=90, concurrent_limit=limit, slot_ttl_sec=900,
                          store=self.store, state_key=GATE_STATE_KEY)

    def _shadow(self):
        p = self.logs / SHADOW_SIGNALS_FILE
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def test_logs_alerts_with_what_the_ledger_needs(self):
        n = _scan_shadow(self.logs, [_row("AAAUSDT", 4.0)], MODEL, self._gate(),
                         min_warmup=60, spread_max_bps=30.0)
        self.assertEqual(n, 1)
        rec = self._shadow()[0]
        self.assertEqual(
            (rec["symbol"], rec["event_ts"], rec["horizon_min"], rec["target_up_pct"], rec["model_run_id"]),
            ("AAAUSDT", 1_700_000_000_000, 15, 0.01, "model_x"),
        )

    def test_same_concurrency_cap_as_the_model(self):
        rows = [_row(f"S{i}USDT", 4.0 + i) for i in range(8)]
        n = _scan_shadow(self.logs, rows, MODEL, self._gate(limit=5), min_warmup=60, spread_max_bps=30.0)
        self.assertEqual(n, 5)

    def test_does_not_share_state_with_the_model_gate(self):
        model_gate = SignalGate(cooldown_sec=90, concurrent_limit=1, slot_ttl_sec=900, store=self.store)
        from types import SimpleNamespace
        self.assertTrue(model_gate.allow(SimpleNamespace(symbol="BUSY", event_ts=1.0)))
        n = _scan_shadow(self.logs, [_row("AAAUSDT", 4.0)], MODEL, self._gate(limit=1),
                         min_warmup=60, spread_max_bps=30.0)
        self.assertEqual(n, 1, "the model's slot blocked the rule")

    def test_a_broken_rule_file_never_breaks_the_scan(self):
        (self.logs / "shadow_rule.json").write_text("{not json")
        self.assertEqual(
            _scan_shadow(self.logs, [_row("AAAUSDT", 4.0)], MODEL, self._gate(),
                         min_warmup=60, spread_max_bps=30.0),
            0,
        )
        save_shadow_rule(self.logs, {"feature": RULE_FEATURE, "threshold": "x"})
        self.assertEqual(
            _scan_shadow(self.logs, [_row("AAAUSDT", 4.0)], MODEL, self._gate(),
                         min_warmup=60, spread_max_bps=30.0),
            0,
        )


if __name__ == "__main__":
    unittest.main()
