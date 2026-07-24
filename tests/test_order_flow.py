"""Tests for taker order-flow ingestion and features (Sprint 3 T3)."""

import json
import tempfile
import unittest
from pathlib import Path

from amber.exchange.normalizer import BybitNormalizer
from amber.features.online import FeatureEngine
from amber.models.features import MODEL_FEATURES
from amber.pipeline.normalize_app import normalize_ws_raw
from amber.storage.state_store import StateStore


def _trade_payload(symbol, ts, side, size):
    return {"topic": f"publicTrade.{symbol}", "type": "snapshot",
            "data": [{"T": ts, "s": symbol, "S": side, "v": str(size), "p": "100"}]}


def _kline_payload(start, confirm=True):
    return {"topic": "kline.1.BTCUSDT", "type": "snapshot", "ts": start + 59_000,
            "data": [{"start": start, "end": start + 60_000, "interval": "1",
                      "open": "100", "high": "101", "low": "99", "close": "100.5",
                      "volume": "42", "confirm": confirm, "timestamp": start + 59_000}]}


class TestTradeParsing(unittest.TestCase):
    def test_trades_from_ws(self):
        trades = BybitNormalizer.trades_from_ws(_trade_payload("BTCUSDT", 1700000000000, "Buy", 5.0))
        self.assertEqual(trades, [("BTCUSDT", 1700000000000, "Buy", 5.0)])

    def test_non_trade_topic_ignored(self):
        self.assertEqual(BybitNormalizer.trades_from_ws({"topic": "kline.1.BTCUSDT", "data": []}), [])


class TestTradeAggregationInNormalize(unittest.TestCase):
    def _write(self, raw_root, payloads):
        target = raw_root / "ws_raw" / "BTCUSDT"
        target.mkdir(parents=True, exist_ok=True)
        with (target / "part-000.jsonl").open("a", encoding="utf-8") as fh:
            for p in payloads:
                fh.write(json.dumps(p) + "\n")

    def _read_norm(self, raw_root):
        path = raw_root / "normalized" / "BTCUSDT" / "part-000.jsonl"
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]

    def test_taker_volume_attached_to_candle(self):
        with tempfile.TemporaryDirectory() as td:
            raw_root = Path(td) / "raw"
            state = StateStore(Path(td) / "state")
            t0 = 1700000000000
            self._write(raw_root, [
                _trade_payload("BTCUSDT", t0 + 1000, "Buy", 10.0),
                _trade_payload("BTCUSDT", t0 + 2000, "Buy", 5.0),
                _trade_payload("BTCUSDT", t0 + 3000, "Sell", 4.0),
                _kline_payload(t0, confirm=True),
            ])
            normalize_ws_raw(raw_root, state)
            row = self._read_norm(raw_root)[0]
            self.assertEqual(row["buy_volume"], 15.0)
            self.assertEqual(row["sell_volume"], 4.0)
            self.assertEqual(row["trade_count"], 3)

    def test_buckets_persist_across_runs(self):
        # trades arrive in run 1, the candle that closes their minute in run 2
        with tempfile.TemporaryDirectory() as td:
            raw_root = Path(td) / "raw"
            state = StateStore(Path(td) / "state")
            t0 = 1700000000000
            self._write(raw_root, [_trade_payload("BTCUSDT", t0 + 1000, "Buy", 7.0)])
            normalize_ws_raw(raw_root, state)  # run 1: only the trade
            self._write(raw_root, [_kline_payload(t0, confirm=True)])
            normalize_ws_raw(raw_root, state)  # run 2: the candle closes
            row = self._read_norm(raw_root)[0]
            self.assertEqual(row["buy_volume"], 7.0)


class TestOrderFlowFeatures(unittest.TestCase):
    def test_features_present_and_imbalance_sign(self):
        eng = FeatureEngine()
        out = {}
        for i in range(30):
            out = eng.update({
                "symbol": "BTCUSDT", "ts": i, "close": 100 + i * 0.01, "volume": 10,
                "buy_volume": 80.0, "sell_volume": 20.0, "trade_count": 50,
            })
        for name in ("taker_imbalance", "cvd_norm_20", "trade_count_z_20"):
            self.assertIn(name, out)
            self.assertIn(name, MODEL_FEATURES)
        self.assertAlmostEqual(out["taker_imbalance"], 0.6, places=6)  # (80-20)/100
        self.assertGreater(out["cvd_norm_20"], 0.5)

    def test_no_trades_gives_zero_imbalance(self):
        eng = FeatureEngine()
        out = eng.update({"symbol": "X", "ts": 0, "close": 100, "volume": 10})
        self.assertEqual(out["taker_imbalance"], 0.0)
        self.assertEqual(out["cvd_norm_20"], 0.0)


if __name__ == "__main__":
    unittest.main()
