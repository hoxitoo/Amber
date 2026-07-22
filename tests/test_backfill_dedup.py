"""REST backfill must fill history *behind* the live WS watermark without
duplicating already-stored candles, and features must recompute correctly."""

import json
import tempfile
import unittest
from pathlib import Path

from amber.exchange.schemas import Candle, FundingRate, OpenInterest, Ticker
from amber.features.compute import compute_batch_features
from amber.pipeline.collector_app import collect_rest_backfill
from amber.storage.parquet_sink import ParquetSink
from amber.storage.state_store import StateStore


class _FakeClient:
    """Returns `n` historical 1m candles ending at `end_ts` (newest last)."""

    def __init__(self, symbol: str, end_ts: int, n: int) -> None:
        self.symbol = symbol
        self.end_ts = end_ts
        self.n = n

    def get_market_snapshot(self, symbol):
        return (
            Ticker(ts=0, symbol=symbol, bid=100.0, ask=100.2, last=100.1),
            OpenInterest(ts=0, symbol=symbol, oi=1000.0),
            FundingRate(ts=0, symbol=symbol, funding=0.0001),
        )

    def get_klines(self, symbol, *, interval="1", limit=200, **kw):
        # include one extra "open" candle at the end (collector drops it)
        out = []
        for i in range(self.n + 1):
            ts = self.end_ts - (self.n - i) * 60_000
            out.append(Candle(ts=ts, symbol=symbol, tf="1m", open=100, high=101, low=99, close=100.5, volume=10))
        return out


def _write_normalized(raw_root: Path, symbol: str, ts_list: list[int]) -> None:
    target = raw_root / "normalized" / symbol
    target.mkdir(parents=True, exist_ok=True)
    with (target / "part-000.jsonl").open("w", encoding="utf-8") as fh:
        for ts in ts_list:
            fh.write(json.dumps({
                "ts": ts, "symbol": symbol, "tf": "1m", "open": 100, "high": 101, "low": 99,
                "close": 100.5, "volume": 10, "bid": 100, "ask": 100.2, "oi": 0, "funding": 0,
                "is_synthetic": False,
            }) + "\n")


class TestBackfillDedup(unittest.TestCase):
    def test_backfill_fills_history_behind_ws_watermark(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_root = root / "raw"
            state = StateStore(root / "state")
            sink = ParquetSink(raw_root)

            now = 1_700_000_000_000 + 200 * 60_000
            # WS already normalized the 5 most recent candles and advanced the watermark
            ws_ts = [now - i * 60_000 for i in range(5)]
            _write_normalized(raw_root, "BTCUSDT", sorted(ws_ts))
            state.set("normalized_watermark", {"last_ts": {"BTCUSDT": max(ws_ts)}})

            # backfill 200 historical candles ending "now"
            client = _FakeClient("BTCUSDT", end_ts=now + 60_000, n=200)
            emitted = collect_rest_backfill(client, sink, state, ["BTCUSDT"], limit=200, raw_root=raw_root)

            # despite the watermark sitting at "now", the ~195 older candles are filled
            self.assertGreater(emitted, 150)
            path = raw_root / "normalized" / "BTCUSDT" / "part-000.jsonl"
            all_ts = [json.loads(x)["ts"] for x in path.read_text(encoding="utf-8").splitlines()]
            # no duplicates
            self.assertEqual(len(all_ts), len(set(all_ts)))
            self.assertGreater(len(all_ts), 150)

    def test_features_recompute_matches_normalized_count(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_root = root / "raw"
            _write_normalized(raw_root, "BTCUSDT", [1_700_000_000_000 + i * 60_000 for i in range(100)])
            out = compute_batch_features(raw_root, root / "features", ["BTCUSDT"])
            self.assertEqual(out["written_rows"], 100)
            # re-run must not duplicate (full rewrite)
            out2 = compute_batch_features(raw_root, root / "features", ["BTCUSDT"])
            self.assertEqual(out2["written_rows"], 100)
            path = root / "features" / "features" / "BTCUSDT" / "part-000.jsonl"
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 100)


if __name__ == "__main__":
    unittest.main()
