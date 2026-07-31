"""Feature computation is incremental but must stay identical to a full recompute.

Recomputing every symbol's whole history each 60s cycle made CPU cost grow with
accumulated data (~4s -> ~38s over a few days at 27 symbols) until it would
exceed the loop interval. Reuse is only safe if it is exactly equivalent — a
divergence here is train/serve skew.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from amber.features.compute import compute_batch_features


def _rows(n: int, start: int = 1_700_000_000_000, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    out, close = [], 100.0
    for i in range(n):
        close *= 1 + rng.gauss(0, 0.003)
        out.append({
            "ts": start + i * 60_000, "symbol": "BTCUSDT", "tf": "1m",
            "open": close, "high": close * 1.002, "low": close * 0.998, "close": close,
            "volume": rng.uniform(1, 100), "bid": close, "ask": close * 1.0001,
            "oi": rng.uniform(900, 1100), "funding": 0.0001,
            "buy_volume": rng.uniform(0, 50), "sell_volume": rng.uniform(0, 50),
            "trade_count": rng.randint(1, 99), "is_synthetic": False,
        })
    return out


def _write(raw: Path, rows: list[dict], mode: str = "w") -> None:
    d = raw / "normalized" / "BTCUSDT"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "part-000.jsonl").open(mode, encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _features(root: Path) -> str:
    return (root / "features" / "features" / "BTCUSDT" / "part-000.jsonl").read_text(encoding="utf-8")


class TestIncrementalEqualsFull(unittest.TestCase):
    def test_appending_in_batches_matches_one_full_pass(self):
        rows = _rows(900)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root / "raw", rows)
            compute_batch_features(root / "raw", root / "features", ["BTCUSDT"])
            full = _features(root)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root / "raw", rows[:300])
            compute_batch_features(root / "raw", root / "features", ["BTCUSDT"])
            _write(root / "raw", rows[300:600], mode="a")
            compute_batch_features(root / "raw", root / "features", ["BTCUSDT"])
            _write(root / "raw", rows[600:], mode="a")
            compute_batch_features(root / "raw", root / "features", ["BTCUSDT"])
            incremental = _features(root)

        self.assertEqual(incremental, full)

    def test_rerun_without_new_candles_is_stable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root / "raw", _rows(300))
            out1 = compute_batch_features(root / "raw", root / "features", ["BTCUSDT"])
            first = _features(root)
            out2 = compute_batch_features(root / "raw", root / "features", ["BTCUSDT"])
            self.assertEqual(_features(root), first)
            self.assertEqual(out1["written_rows"], out2["written_rows"])
            self.assertEqual(len(first.splitlines()), 300)

    def test_history_inserted_behind_triggers_full_recompute(self):
        """REST backfill fills candles *behind* the live stream: the stored
        prefix is then wrong and must not be reused."""
        later = _rows(300, start=1_700_000_000_000 + 300 * 60_000, seed=3)
        earlier = _rows(300, start=1_700_000_000_000, seed=9)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root / "raw", later)
            compute_batch_features(root / "raw", root / "features", ["BTCUSDT"])
            _write(root / "raw", earlier, mode="a")  # backfill lands behind
            compute_batch_features(root / "raw", root / "features", ["BTCUSDT"])
            got = _features(root)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root / "raw", earlier + later)
            compute_batch_features(root / "raw", root / "features", ["BTCUSDT"])
            expected = _features(root)

        self.assertEqual(got, expected)
        self.assertEqual(len(got.splitlines()), 600)

    def test_feature_spec_bump_forces_recompute(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root / "raw", _rows(300))
            compute_batch_features(root / "raw", root / "features", ["BTCUSDT"])
            meta = root / "features" / "features" / "BTCUSDT" / "meta.json"
            meta.write_text(json.dumps({"spec_version": "v0", "rows": 300, "last_ts": 0}), encoding="utf-8")

            compute_batch_features(root / "raw", root / "features", ["BTCUSDT"])
            self.assertEqual(len(_features(root).splitlines()), 300)
            self.assertEqual(json.loads(meta.read_text())["spec_version"], "v4")

    def test_truncated_feature_file_falls_back_to_full_recompute(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root / "raw", _rows(300))
            compute_batch_features(root / "raw", root / "features", ["BTCUSDT"])
            good = _features(root)

            f = root / "features" / "features" / "BTCUSDT" / "part-000.jsonl"
            f.write_text("\n".join(good.splitlines()[:120]) + "\n", encoding="utf-8")

            compute_batch_features(root / "raw", root / "features", ["BTCUSDT"])
            self.assertEqual(_features(root), good)


if __name__ == "__main__":
    unittest.main()
