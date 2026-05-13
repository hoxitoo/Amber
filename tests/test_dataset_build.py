import tempfile
import unittest
from pathlib import Path

from amber.datasets.build import build_dataset


class TestDatasetBuild(unittest.TestCase):
    def test_build_dataset_raises_on_invalid_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            features_root = root / "features_root"
            feat_dir = features_root / "features" / "BTCUSDT"
            feat_dir.mkdir(parents=True)
            (feat_dir / "part-000.jsonl").write_text('{"ts": 1}\n{bad json}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid JSONL"):
                build_dataset(features_root=features_root, datasets_root=root / "datasets", symbols=["BTCUSDT"])

    def test_build_dataset_rejects_invalid_params(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            features_root = root / "features_root"
            with self.assertRaisesRegex(ValueError, "horizon_steps must be > 0"):
                build_dataset(features_root=features_root, datasets_root=root / "datasets", symbols=["BTCUSDT"], horizon_steps=0)
            with self.assertRaisesRegex(ValueError, "must be > 0"):
                build_dataset(features_root=features_root, datasets_root=root / "datasets", symbols=["BTCUSDT"], up_pct=0, down_pct=0.01)

    def test_build_dataset_adaptive_thresholds(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            features_root = root / "features_root"
            feat_dir = features_root / "features" / "BTCUSDT"
            feat_dir.mkdir(parents=True)
            rows = []
            for i in range(8):
                rows.append({"ts": i, "mid_price": 100 + i * 0.1, "ret_1": 0.001 * i, "vol_z_20": 0.1, "is_synthetic": False})
            (feat_dir / "part-000.jsonl").write_text("\n".join(__import__('json').dumps(r) for r in rows)+"\n", encoding="utf-8")
            out = build_dataset(
                features_root=features_root,
                datasets_root=root / "datasets",
                symbols=["BTCUSDT"],
                horizon_steps=2,
                adaptive_thresholds=True,
                threshold_k=2.5,
                threshold_vol_window=3,
                threshold_floor=0.003,
                threshold_cap=0.01,
            )
            self.assertGreater(out["rows"], 0)
            ds = sorted((root / "datasets").glob("dataset_*/dataset.jsonl"))[-1]
            first = __import__('json').loads(ds.read_text(encoding="utf-8").splitlines()[0])
            self.assertGreaterEqual(first["up_pct"], 0.003)
            self.assertLessEqual(first["up_pct"], 0.01)



if __name__ == "__main__":
    unittest.main()
