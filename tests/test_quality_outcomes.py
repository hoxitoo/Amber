"""Tests for the honest rolling-AUC outcome join in the quality report."""

import json
import tempfile
import unittest
from pathlib import Path

from amber.monitoring.quality_report import build_quality_report


def _write_candles(raw_root: Path, symbol: str, t0: int, closes: list[float]) -> None:
    target = raw_root / "normalized" / symbol
    target.mkdir(parents=True, exist_ok=True)
    with (target / "part-000.jsonl").open("w", encoding="utf-8") as fh:
        for i, close in enumerate(closes):
            row = {
                "ts": t0 + i * 60_000,
                "symbol": symbol,
                "close": close,
                "high": close * 1.001,
                "low": close * 0.999,
            }
            fh.write(json.dumps(row) + "\n")


def _signal(symbol: str, event_ts_ms: int, p_up: float, target_pct: float, horizon: int) -> str:
    return json.dumps(
        {
            "symbol": symbol,
            "event_ts": event_ts_ms,
            "prob_up_calibrated": p_up,
            "prob_down_calibrated": 1 - p_up,
            "target_up_pct": target_pct,
            "horizon_min": horizon,
        }
    )


class TestQualityOutcomes(unittest.TestCase):
    def test_auc_uses_real_outcomes_not_pseudo_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_root = root / "raw"
            t0 = 1700000000000
            # price jumps +1% at candle 3 and stays flat after
            closes = [100.0, 100.0, 100.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0]
            _write_candles(raw_root, "BTCUSDT", t0, closes)

            signals = [
                # confident pump signal at t0 -> hit (+1% within 5 candles)
                _signal("BTCUSDT", t0, p_up=0.9, target_pct=0.005, horizon=5),
                # confident pump signal after the move -> no further move -> miss
                _signal("BTCUSDT", t0 + 4 * 60_000, p_up=0.85, target_pct=0.005, horizon=5),
            ] * 12  # enough points for the AUC window minimum
            signals_path = root / "signals.jsonl"
            signals_path.write_text("\n".join(signals) + "\n", encoding="utf-8")

            rep = build_quality_report(signals_path, raw_root=raw_root)
            self.assertEqual(rep["auc_confirmed_outcomes"], 24)
            self.assertIsNotNone(rep["rolling_auc"])
            # first signal hit, second missed, probabilities 0.9 vs 0.85 -> AUC 1.0
            self.assertGreater(rep["rolling_auc"], 0.99)

    def test_unresolved_horizon_is_not_counted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_root = root / "raw"
            t0 = 1700000000000
            _write_candles(raw_root, "BTCUSDT", t0, [100.0, 100.0, 100.0])
            # horizon extends past available data -> outcome unknown
            signals_path = root / "signals.jsonl"
            signals_path.write_text(_signal("BTCUSDT", t0, 0.9, 0.005, horizon=10) + "\n", encoding="utf-8")
            rep = build_quality_report(signals_path, raw_root=raw_root)
            self.assertEqual(rep["auc_confirmed_outcomes"], 0)
            self.assertEqual(rep["auc_unconfirmed_outcomes"], 1)
            self.assertIsNone(rep["rolling_auc"])

    def test_without_raw_root_auc_is_none_not_fabricated(self):
        with tempfile.TemporaryDirectory() as td:
            signals_path = Path(td) / "signals.jsonl"
            rows = [
                json.dumps({"prob_up_calibrated": 0.8, "prob_down_calibrated": 0.2})
                for _ in range(50)
            ]
            signals_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            rep = build_quality_report(signals_path)
            self.assertIsNone(rep["rolling_auc"])
            self.assertEqual(rep["psi"]["level"], "unavailable")


if __name__ == "__main__":
    unittest.main()
