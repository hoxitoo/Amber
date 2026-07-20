import json
import tempfile
import unittest
from pathlib import Path

from amber.pipeline.normalize_app import normalize_ws_raw
from amber.storage.state_store import StateStore


def _kline(start: int, confirm: bool = True, close: str = "100.5") -> dict:
    return {
        "topic": "kline.1.BTCUSDT",
        "type": "snapshot",
        "ts": start + 30_000,
        "data": [
            {
                "start": start,
                "end": start + 60_000,
                "interval": "1",
                "open": "100",
                "high": "101",
                "low": "99",
                "close": close,
                "volume": "42",
                "confirm": confirm,
                "timestamp": start + 30_000,
            }
        ],
    }


def _ticker(ts: int) -> dict:
    return {
        "topic": "tickers.BTCUSDT",
        "type": "snapshot",
        "ts": ts,
        "data": {
            "symbol": "BTCUSDT",
            "lastPrice": "100.1",
            "bid1Price": "100.0",
            "ask1Price": "100.2",
            "fundingRate": "0.0001",
            "openInterest": "123456",
        },
    }


class TestNormalizePipeline(unittest.TestCase):
    def _write_raw(self, raw_root: Path, payloads: list[dict]) -> Path:
        target = raw_root / "ws_raw" / "BTCUSDT"
        target.mkdir(parents=True, exist_ok=True)
        path = target / "part-000.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            for p in payloads:
                fh.write(json.dumps(p) + "\n")
        return path

    def _read_normalized(self, raw_root: Path) -> list[dict]:
        path = raw_root / "normalized" / "BTCUSDT" / "part-000.jsonl"
        if not path.exists():
            return []
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]

    def test_confirm_only_market_state_and_gap_fill(self):
        with tempfile.TemporaryDirectory() as td:
            raw_root = Path(td) / "raw"
            state = StateStore(Path(td) / "state")
            t0 = 1700000000000
            self._write_raw(
                raw_root,
                [
                    _ticker(t0),
                    _kline(t0, confirm=False),  # in-progress update must be skipped
                    _kline(t0, confirm=True),
                    _kline(t0 + 180_000, confirm=True),  # 3m later -> 2 synthetic rows
                ],
            )
            written = normalize_ws_raw(raw_root, state)
            rows = self._read_normalized(raw_root)
            self.assertEqual(written, 4)  # 2 real + 2 synthetic
            self.assertEqual([r["ts"] for r in rows], [t0, t0 + 60_000, t0 + 120_000, t0 + 180_000])
            self.assertEqual([r["is_synthetic"] for r in rows], [False, True, True, False])
            # ticker state must reach the candle row
            self.assertEqual(rows[0]["bid"], 100.0)
            self.assertEqual(rows[0]["ask"], 100.2)
            self.assertEqual(rows[0]["oi"], 123456.0)
            self.assertEqual(rows[0]["funding"], 0.0001)

    def test_rerun_is_idempotent_and_incremental(self):
        with tempfile.TemporaryDirectory() as td:
            raw_root = Path(td) / "raw"
            state = StateStore(Path(td) / "state")
            t0 = 1700000000000
            self._write_raw(raw_root, [_kline(t0)])
            self.assertEqual(normalize_ws_raw(raw_root, state), 1)
            # rerun with no new data -> nothing written
            self.assertEqual(normalize_ws_raw(raw_root, state), 0)
            self.assertEqual(len(self._read_normalized(raw_root)), 1)
            # append one new candle -> exactly one new row
            self._write_raw(raw_root, [_kline(t0 + 60_000)])
            self.assertEqual(normalize_ws_raw(raw_root, state), 1)
            self.assertEqual(len(self._read_normalized(raw_root)), 2)

    def test_duplicate_candles_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as td:
            raw_root = Path(td) / "raw"
            state = StateStore(Path(td) / "state")
            t0 = 1700000000000
            self._write_raw(raw_root, [_kline(t0), _kline(t0), _kline(t0)])
            self.assertEqual(normalize_ws_raw(raw_root, state), 1)


if __name__ == "__main__":
    unittest.main()
