"""Guards against bugs that do not raise — the kind that ran for ten days.

Signal duplication went unnoticed because nothing crashed: the numbers just
quietly described something other than what they claimed. Each test here pins an
invariant whose violation would be equally silent.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from amber.common.jsonl import read_last, read_tail


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


class TestPsiCoversEverySymbol(unittest.TestCase):
    """PSI pooled rows from all symbols then trimmed the pool to `window`,
    keeping only the last symbol alphabetically — a universe-wide metric that
    silently described one coin."""

    def _project(self, td: str, n_symbols: int = 5, rows_per_symbol: int = 300):
        root = Path(td)
        from amber.models.train import _feature_quantiles

        rng = random.Random(4)
        train = [{"ret_1": rng.gauss(0, 0.01), "vol_z_20": rng.gauss(0, 1)} for _ in range(2000)]
        ref = _feature_quantiles(train)
        md = root / "models" / "model_20260101T000000Z_aa"
        md.mkdir(parents=True)
        (md / "model.json").write_text(
            json.dumps({"model_type": "lightgbm_dual_v1", "heads": {}, "train_reference": ref}), encoding="utf-8"
        )
        for i in range(n_symbols):
            # One symbol is wildly out of distribution; pooling must not hide it.
            shift = 5.0 if i == 0 else 0.0
            _write(
                root / "features" / "features" / f"AAA{i}USDT" / "part-000.jsonl",
                [{"ret_1": rng.gauss(shift, 0.01), "vol_z_20": rng.gauss(shift, 1)} for _ in range(rows_per_symbol)],
            )
        return root

    def test_every_symbol_contributes(self):
        from amber.monitoring.quality_report import _feature_psi

        with tempfile.TemporaryDirectory() as td:
            root = self._project(td)
            res = _feature_psi(root / "models", root / "features", window=100)
            self.assertEqual(res["reason"], "ok")
            self.assertEqual(res["symbols"], 5, "PSI did not read every symbol")

    def test_an_outlier_symbol_is_not_averaged_away_by_trimming(self):
        """With the old trimming the first symbol (the drifted one) was dropped
        entirely, so drift went unreported."""
        from amber.monitoring.quality_report import _feature_psi

        with tempfile.TemporaryDirectory() as td:
            root = self._project(td)
            res = _feature_psi(root / "models", root / "features", window=100)
            self.assertGreater(res["max_psi"], 0.1, "drifted symbol left no trace in the pooled PSI")


class TestInferenceKeepsModelSerialisable(unittest.TestCase):
    """Scoring used to stash a live Booster inside the model dict, so any code
    that scored before saving would break the retrain with a TypeError."""

    def _model(self):
        import lightgbm as lgb
        import numpy as np

        rng = random.Random(1)
        x = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(200)]
        y = [1 if r[0] > 0 else 0 for r in x]
        booster = lgb.train(
            {"objective": "binary", "verbosity": -1, "num_leaves": 5},
            lgb.Dataset(np.asarray(x), label=np.asarray(y, dtype=float)),
            num_boost_round=5,
        )
        return {
            "features": ["f0", "f1", "f2"],
            "heads": {"pump": {"type": "lightgbm", "booster": booster.model_to_string(), "label_rate": 0.5}},
        }

    def test_model_still_serialises_after_scoring(self):
        from amber.models.infer import infer_row_prob

        model = self._model()
        infer_row_prob(model, {"f0": 0.5, "f1": 0.1, "f2": 0.2}, target="pump")
        json.dumps(model)  # must not raise

    def test_batch_and_row_scoring_agree(self):
        from amber.models.importance import _predict_matrix
        from amber.models.infer import infer_row_prob

        model = self._model()
        row = {"f0": 0.5, "f1": 0.1, "f2": 0.2}
        self.assertAlmostEqual(
            infer_row_prob(model, row, target="pump"),
            _predict_matrix(model, "pump", [[0.5, 0.1, 0.2]])[0],
            places=12,
        )


class TestBoundedReads(unittest.TestCase):
    def test_tail_returns_the_newest_rows(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "f.jsonl"
            _write(p, [{"i": i} for i in range(5000)])
            tail = read_tail(p, 10)
            self.assertEqual(len(tail), 10)
            self.assertEqual(tail[-1]["i"], 4999)

    def test_read_last_survives_a_truncated_final_line(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "f.jsonl"
            p.write_text('{"i": 1}\n{"i": 2}\n{"i": 3, "x": "unterm', encoding="utf-8")
            self.assertEqual(read_last(p)["i"], 2)

    def test_missing_file_is_safe(self):
        self.assertEqual(read_tail(Path("/nonexistent/x.jsonl"), 5), [])
        self.assertIsNone(read_last(Path("/nonexistent/x.jsonl")))


class TestRetrainMemoryDiscipline(unittest.TestCase):
    def test_diagnostics_do_not_reload_the_dataset(self):
        """The dataset is hundreds of MB; the importance diagnostic must reuse
        rows already in memory rather than materialise them again."""
        src = (Path(__file__).resolve().parents[1] / "amber" / "pipeline" / "train_app.py").read_text(encoding="utf-8")
        self.assertEqual(
            src.count("load_latest_dataset_rows(datasets_root)"), 1,
            "train_app loads the full dataset more than once",
        )


if __name__ == "__main__":
    unittest.main()
