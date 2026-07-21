"""Tests for Sprint-1 audit fixes: atomic state, single-instance lock,
warm-up gating, PR-AUC + reliability, universe logging."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from amber.common.audit_log import log_universe
from amber.common.locks import AlreadyRunning, SingleInstanceLock
from amber.datasets.build import build_dataset
from amber.models.eval import _reliability_curve, _safe_pr_auc
from amber.storage.state_store import StateStore


class TestAtomicStateStore(unittest.TestCase):
    def test_set_then_get_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            s = StateStore(Path(td))
            s.set("wm", {"last_ts": {"BTCUSDT": 123}})
            self.assertEqual(s.get("wm"), {"last_ts": {"BTCUSDT": 123}})

    def test_set_leaves_no_temp_files_and_overwrites(self):
        with tempfile.TemporaryDirectory() as td:
            s = StateStore(Path(td))
            s.set("k", {"v": 1})
            s.set("k", {"v": 2})
            self.assertEqual(s.get("k"), {"v": 2})
            leftovers = [p for p in Path(td).iterdir() if p.suffix == ".tmp"]
            self.assertEqual(leftovers, [])


class TestSingleInstanceLock(unittest.TestCase):
    def test_second_acquire_blocks_while_held(self):
        with tempfile.TemporaryDirectory() as td:
            lock_dir = Path(td)
            with SingleInstanceLock(lock_dir, "stage"):
                with self.assertRaises(AlreadyRunning):
                    SingleInstanceLock(lock_dir, "stage").acquire()
            # released after context exit -> can acquire again
            with SingleInstanceLock(lock_dir, "stage"):
                pass

    def test_stale_lock_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as td:
            lock_dir = Path(td)
            # simulate a dead-owner lock: write a PID that is not alive
            lock_dir.mkdir(exist_ok=True)
            dead_pid = 2**31 - 1  # implausible, not alive
            (lock_dir / "stage.lock").write_text(str(dead_pid), encoding="utf-8")
            with SingleInstanceLock(lock_dir, "stage"):
                owner = int((lock_dir / "stage.lock").read_text(encoding="utf-8"))
                self.assertEqual(owner, os.getpid())


class TestWarmupGating(unittest.TestCase):
    def _write_features(self, root: Path, obs_start: int, n: int) -> Path:
        feat = root / "features" / "features" / "BTCUSDT"
        feat.mkdir(parents=True)
        with (feat / "part-000.jsonl").open("w", encoding="utf-8") as fh:
            for i in range(n):
                fh.write(json.dumps({
                    "symbol": "BTCUSDT", "ts": i, "mid_price": 100 + i * 0.1,
                    "ret_1": 0.001, "obs": obs_start + i, "is_synthetic": False,
                }) + "\n")
        return root / "features"

    def test_warmup_rows_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            features_root = self._write_features(root, obs_start=1, n=80)
            # with min_warmup_bars=60 only rows with obs>=60 (and full forward window) qualify
            build_dataset(
                features_root=features_root, datasets_root=root / "ds",
                symbols=["BTCUSDT"], horizon_steps=2, min_warmup_bars=60,
            )
            ds = sorted((root / "ds").glob("dataset_*/dataset.jsonl"))[-1]
            rows = [json.loads(x) for x in ds.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(rows)
            self.assertTrue(all(r["obs"] >= 60 for r in rows))

    def test_default_no_warmup_filter_keeps_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            features_root = self._write_features(root, obs_start=1, n=20)
            out = build_dataset(
                features_root=features_root, datasets_root=root / "ds",
                symbols=["BTCUSDT"], horizon_steps=2,  # default min_warmup_bars=0
            )
            self.assertGreater(out["rows"], 0)


class TestPRAUCandReliability(unittest.TestCase):
    def test_pr_auc_perfect_ranking(self):
        y = [0, 0, 0, 1, 1]
        p = [0.1, 0.2, 0.3, 0.8, 0.9]
        self.assertAlmostEqual(_safe_pr_auc(y, p), 1.0, places=6)

    def test_pr_auc_none_on_single_class(self):
        self.assertIsNone(_safe_pr_auc([0, 0, 0], [0.1, 0.2, 0.3]))

    def test_reliability_curve_bins(self):
        y = [0, 0, 1, 1]
        p = [0.05, 0.15, 0.85, 0.95]
        curve = _reliability_curve(y, p, bins=10)
        self.assertTrue(all("mean_pred" in b and "observed" in b for b in curve))
        self.assertEqual(sum(b["n"] for b in curve), 4)


class TestUniverseLog(unittest.TestCase):
    def test_log_universe_appends(self):
        with tempfile.TemporaryDirectory() as td:
            logs = Path(td)
            log_universe(logs, "collector", ["ETHUSDT", "BTCUSDT"], extra={"run_id": "r1"})
            rows = (logs / "universe.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            rec = json.loads(rows[0])
            self.assertEqual(rec["symbols"], ["BTCUSDT", "ETHUSDT"])  # sorted
            self.assertEqual(rec["count"], 2)
            self.assertEqual(rec["stage"], "collector")
            self.assertEqual(rec["run_id"], "r1")


if __name__ == "__main__":
    unittest.main()
