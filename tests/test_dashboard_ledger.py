"""The ledger panel renders on the real dashboard, empty and populated.

The dashboard has no other render test; a panel that raises takes the whole
page down with it, which is how the owner would first find out.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

try:
    from streamlit.testing.v1 import AppTest
except Exception:  # pragma: no cover - dashboard deps are optional
    AppTest = None


def _ledger_rows():
    rows = []
    for e in range(40):
        g = 0.01 if e % 3 else -0.01
        rows.append({
            "source": "model" if e % 2 else "rule", "status": "ok", "symbol": "AAAUSDT",
            "horizon": 15, "event_ts": 1_700_000_000_000 + e * 1_800_000, "move_hit": 1,
            "momentum_net": g - 0.0009, "fade_net": -g - 0.0009,
        })
    return rows


@unittest.skipIf(AppTest is None, "streamlit not installed")
class TestLedgerPanel(unittest.TestCase):
    def _render(self, with_ledger: bool, with_ignition: bool = False):
        tmp = Path(tempfile.mkdtemp())
        shutil.copytree(REPO / "config", tmp / "config")
        if with_ledger:
            logs = tmp / "data" / "logs"
            logs.mkdir(parents=True)
            (logs / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _ledger_rows()))
        if with_ignition:
            logs = tmp / "data" / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            rec = {"event_ts": 1_700_000_000_000, "symbol": "ETHUSDT", "score": 0.4, "prob": 0.093,
                   "threshold": 0.3, "coin_base": 0.012, "alert": 1, "notify": 1, "horizon_min": 30,
                   "target_up_pct": 0.01, "factors": ["размах цены за 4 ч"]}
            (logs / "ignition_calm.jsonl").write_text(json.dumps(rec, ensure_ascii=False) + "\n")
            (logs / "ignition_ledger.jsonl").write_text(
                json.dumps({**rec, "status": "ok", "move_hit": 1}) + "\n")
            from amber.signals.ignition_live import save_summary, summarize_ignition

            save_summary(logs, summarize_ignition(logs))
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            at = AppTest.from_file(str(REPO / "amber" / "dashboard" / "app.py"), default_timeout=60)
            at.run()
        finally:
            os.chdir(cwd)
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at

    def test_empty_ledger_says_so(self):
        at = self._render(with_ledger=False)
        self.assertTrue(any("Журнал пуст" in i.value for i in at.info))

    def test_ignition_panel_shows_the_warning_as_described(self):
        at = self._render(with_ledger=False, with_ignition=True)
        tables = [df.value for df in at.dataframe]
        ign = next((t for t in tables if "P(ход)" in t.columns), None)
        self.assertIsNotNone(ign, "ignition table not rendered")
        self.assertEqual(ign.iloc[0]["символ"], "ETHUSDT")
        self.assertIn("размах цены за 4 ч", ign.iloc[0]["факторы"])

    def test_populated_ledger_shows_both_sources_and_both_rules(self):
        at = self._render(with_ledger=True)
        tables = [df.value for df in at.dataframe]
        ledger = next((t for t in tables if "стратегия" in t.columns), None)
        self.assertIsNotNone(ledger, "ledger table not rendered")
        self.assertEqual(len(ledger), 4)
        self.assertEqual(set(ledger["источник"]), {"модель", "правило ATR"})


if __name__ == "__main__":
    unittest.main()
