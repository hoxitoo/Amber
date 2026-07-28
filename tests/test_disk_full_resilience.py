"""Regression tests for the 2026-07 disk-full wedge (audit C1-C4).

A full disk took the whole box down and it could not recover on its own:
cleanup ran only after the writes that were failing, a partial dataset write
became the permanent "latest" dataset, and retention only ran as the tail of a
successful training run.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from amber.datasets.build import _write_jsonl
from amber.models.dataset_io import read_jsonl_strict
from amber.pipeline.normalize_app import OFFSETS_STATE_KEY, WATERMARK_STATE_KEY, normalize_ws_raw
from amber.storage.state_store import StateStore


def _kline(symbol: str, ts: int, close: float) -> dict:
    return {
        "topic": f"kline.1.{symbol}",
        "data": [{
            "start": ts, "interval": "1", "open": close, "high": close,
            "low": close, "close": close, "volume": 10, "confirm": True,
        }],
    }


def _write_ws_raw(raw_root: Path, symbol: str, part: str, payloads: list[dict]) -> Path:
    d = raw_root / "ws_raw" / symbol
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{part}.jsonl"
    with f.open("a", encoding="utf-8") as fh:
        for p in payloads:
            fh.write(json.dumps(p) + "\n")
    return f


class TestC1CleanupRunsBeforeWrites(unittest.TestCase):
    def test_consumed_ws_raw_is_reclaimed_even_when_writes_fail(self):
        """The disk reclaim must not sit behind the write that ENOSPC kills."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw"
            state = StateStore(root / "state")

            old = _write_ws_raw(raw, "BTCUSDT", "part-2026072800", [_kline("BTCUSDT", 1, 100.0)])
            _write_ws_raw(raw, "BTCUSDT", "part-2026072801", [_kline("BTCUSDT", 60_000, 101.0)])
            # mark the rotated file fully consumed
            state.set(OFFSETS_STATE_KEY, {"BTCUSDT/part-2026072800.jsonl": old.stat().st_size})

            with mock.patch(
                "amber.storage.parquet_sink.ParquetSink.write_records",
                side_effect=OSError(28, "No space left on device"),
            ):
                with self.assertRaises(OSError):
                    normalize_ws_raw(raw, state)

            self.assertFalse(old.exists(), "consumed ws_raw file was not reclaimed before the failing write")


class TestC2NoProgressLossOnFailedWrite(unittest.TestCase):
    def test_offset_and_watermark_do_not_advance_past_unwritten_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw"
            state = StateStore(root / "state")
            _write_ws_raw(raw, "BTCUSDT", "part-2026072800", [_kline("BTCUSDT", 60_000, 100.0)])

            with mock.patch(
                "amber.storage.parquet_sink.ParquetSink.write_records",
                side_effect=OSError(28, "No space left on device"),
            ):
                with self.assertRaises(OSError):
                    normalize_ws_raw(raw, state)

            # nothing was written, so nothing may be marked as consumed
            self.assertEqual(state.get(OFFSETS_STATE_KEY).get("BTCUSDT/part-2026072800.jsonl", 0), 0)
            self.assertEqual(state.get(WATERMARK_STATE_KEY).get("last_ts", {}).get("BTCUSDT", 0), 0)

            # once the disk is healthy again the candle is picked up, not skipped
            written = normalize_ws_raw(raw, state)
            self.assertEqual(written, 1)
            self.assertEqual(state.get(WATERMARK_STATE_KEY)["last_ts"]["BTCUSDT"], 60_000)


class TestC3AtomicDatasetWrite(unittest.TestCase):
    def test_failed_write_leaves_previous_dataset_intact(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dataset.jsonl"
            _write_jsonl(path, [{"a": 1}, {"a": 2}])
            good = path.read_text(encoding="utf-8")

            with mock.patch("json.dumps", side_effect=OSError(28, "No space left on device")):
                with self.assertRaises(OSError):
                    _write_jsonl(path, [{"a": 3}])

            self.assertEqual(path.read_text(encoding="utf-8"), good)
            self.assertFalse(path.with_suffix(".jsonl.tmp").exists(), "temp file left behind")

    def test_truncated_final_line_is_dropped_not_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dataset.jsonl"
            path.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "b": "unterm', encoding="utf-8")
            rows = read_jsonl_strict(path)
            self.assertEqual(rows, [{"a": 1}, {"a": 2}])

    def test_corruption_before_the_last_line_still_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dataset.jsonl"
            path.write_text('{"a": 1}\n{bad json}\n{"a": 3}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_jsonl_strict(path)


class TestC4RetentionRunsUnconditionally(unittest.TestCase):
    def test_sweep_prunes_without_a_successful_training_run(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "run_pipeline_loop", Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline_loop.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            datasets = root / "datasets"
            for name in ("dataset_001", "dataset_002", "dataset_003"):
                (datasets / name).mkdir(parents=True)
                (datasets / name / "dataset.jsonl").write_text("{}\n", encoding="utf-8")

            cfg = {"storage": {
                "datasets_dir": str(datasets), "models_dir": str(root / "models"),
                "raw_dir": str(root / "raw"), "state_dir": str(root / "state"), "keep_runs": 2,
            }}
            mod._retention_sweep(cfg)

            remaining = sorted(p.name for p in datasets.iterdir())
            self.assertEqual(remaining, ["dataset_002", "dataset_003"])

    def test_free_bytes_reports_a_positive_number(self):
        from amber.common.retention import free_bytes

        with tempfile.TemporaryDirectory() as td:
            self.assertGreater(free_bytes(Path(td) / "does" / "not" / "exist"), 0)

    def test_datasets_get_a_tighter_budget_than_models(self):
        """Datasets are the biggest artifact and are regenerable, so they must
        not inherit the model retention count (that is what reached 53 GB)."""
        from amber.common.retention import dataset_keep

        self.assertEqual(dataset_keep({"keep_runs": 5}), 2)  # not 5
        self.assertEqual(dataset_keep({"keep_dataset_runs": 3}), 3)
        self.assertEqual(dataset_keep({"keep_dataset_runs": 0}), 1)  # never zero
        self.assertEqual(dataset_keep({"keep_dataset_runs": "bad"}), 2)

    def test_sweep_applies_the_dataset_budget_not_keep_runs(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "run_pipeline_loop", Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline_loop.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            datasets = root / "datasets"
            for i in range(6):
                (datasets / f"dataset_{i:03d}").mkdir(parents=True)
                (datasets / f"dataset_{i:03d}" / "dataset.jsonl").write_text("{}\n", encoding="utf-8")

            mod._retention_sweep({"storage": {
                "datasets_dir": str(datasets), "models_dir": str(root / "models"),
                "raw_dir": str(root / "raw"), "state_dir": str(root / "state"),
                "keep_runs": 5, "keep_dataset_runs": 2,
            }})
            self.assertEqual(sorted(p.name for p in datasets.iterdir()), ["dataset_004", "dataset_005"])


if __name__ == "__main__":
    unittest.main()
