"""Tests for Sprint-2 audit fixes: coherent probabilities, clipping, lagged
adaptive threshold, entry-lag backtest, per-regime eval, config hashing,
plain-text telegram pushes."""

import json
import tempfile
import unittest
from pathlib import Path

from amber.backtest.backtester import _signal_replay
from amber.common.manifest import ArtifactManifest, write_manifest
from amber.datasets.build import _adaptive_threshold
from amber.models.eval import regime_metrics
from amber.models.infer import infer_row_prob
from amber.models.train import _apply_clip, _clip_bounds
from amber.signals.scorer import coherent_pump_dump


class TestCoherentProbabilities(unittest.TestCase):
    def test_overlapping_heads_are_normalized(self):
        up, down = coherent_pump_dump(0.8, 0.6)
        self.assertAlmostEqual(up + down, 1.0)
        self.assertAlmostEqual(up / down, 0.8 / 0.6, places=6)

    def test_valid_distribution_untouched(self):
        self.assertEqual(coherent_pump_dump(0.3, 0.2), (0.3, 0.2))


class TestClipping(unittest.TestCase):
    def test_bounds_and_apply(self):
        # ret_1 is feature index 0; one extreme outlier must be clipped away
        from amber.models.features import MODEL_FEATURES

        x = [[0.001 * i] + [0.0] * (len(MODEL_FEATURES) - 1) for i in range(100)]
        x[50][0] = 999.0
        bounds = _clip_bounds(x)
        self.assertIn("ret_1", bounds)
        lo, hi = bounds["ret_1"]
        self.assertLess(hi, 999.0)
        clipped = _apply_clip(x, bounds)
        self.assertLessEqual(max(row[0] for row in clipped), hi)

    def test_infer_applies_model_clip_bounds(self):
        model = {
            "features": ["ret_1"],
            "clip_bounds": {"ret_1": [-0.01, 0.01]},
            "heads": {"pump": {"type": "logreg", "weights": {"ret_1": 100.0}, "bias": 0.0}},
        }
        p_extreme = infer_row_prob(model, {"ret_1": 999.0}, target="pump")
        p_at_bound = infer_row_prob(model, {"ret_1": 0.01}, target="pump")
        self.assertAlmostEqual(p_extreme, p_at_bound, places=9)


class TestLaggedAdaptiveThreshold(unittest.TestCase):
    def test_current_bar_excluded_from_vol_window(self):
        rets = [0.001] * 10 + [0.5]  # huge move on the current bar
        thr_now = _adaptive_threshold(rets, idx=10, k=2.5, window=5, floor=0.0001, cap=10.0)
        thr_prev = _adaptive_threshold(rets, idx=9, k=2.5, window=5, floor=0.0001, cap=10.0)
        # the spike at idx=10 must not inflate its own bar's threshold
        self.assertLess(thr_now, 0.01)
        self.assertAlmostEqual(thr_now, thr_prev, places=12)


class TestEntryLagBacktest(unittest.TestCase):
    def test_outcome_booked_from_next_bar(self):
        model = {"heads": {"pump": {"type": "constant", "prob": 0.9}, "dump": {"type": "constant", "prob": 0.05}}}
        thresholds = {
            "pump_prob_calibrated_min": 0.5,
            "dump_prob_calibrated_min": 0.5,
            "directional_score_min": 0.1,
            "spread_bps_max": 100.0,
        }
        rows = [
            # decision bar: its own window would be a TP...
            {"symbol": "A", "up_hit": 1, "down_hit": 0, "up_pct": 0.01, "down_pct": 0.01,
             "horizon_steps": 5, "spread_bps": 1.0},
            # ...but entry is next bar, whose window times out
            {"symbol": "A", "up_hit": 0, "down_hit": 0, "up_pct": 0.01, "down_pct": 0.01,
             "horizon_steps": 5, "spread_bps": 1.0},
        ]
        pnls, counts, extra = _signal_replay(rows, model, {"method": "identity"}, thresholds, cost=0.0)
        self.assertEqual(extra["entry_lag_bars"], 1)
        self.assertEqual(counts, {"TP": 0, "SL": 0, "Timeout": 1})
        self.assertEqual(extra["trades_pump"], 1)


class TestRegimeMetrics(unittest.TestCase):
    def test_four_buckets_cover_all_rows(self):
        rows = []
        probs = []
        y = []
        for i in range(40):
            rows.append({"ret_60": 0.02 if i % 2 else 0.001, "bb_width_20": 0.03 if i % 4 < 2 else 0.001})
            probs.append(0.8 if i % 3 == 0 else 0.2)
            y.append(1 if i % 3 == 0 else 0)
        out = regime_metrics(rows, probs, y)
        self.assertEqual(sum(int(m["rows"]) for m in out.values()), 40)
        for name in out:
            trend, vol = name.split("_")
            self.assertIn(trend, ("trend", "range"))
            self.assertIn(vol, ("hivol", "lovol"))
            self.assertIn("base_rate_up", out[name])


class TestConfigHashesInManifest(unittest.TestCase):
    def test_write_manifest_embeds_hash_keys(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            manifest = ArtifactManifest(
                run_id="r1", artifact_type="test", artifact_version="v1",
                created_at="now", config_ref="config/amber.yaml",
                feature_spec_ref="config/features.yaml", metadata={"x": 1},
            )
            write_manifest(path, manifest)
            data = json.loads(path.read_text(encoding="utf-8"))
            for key in ("config_sha256", "features_sha256", "thresholds_sha256"):
                self.assertIn(key, data["metadata"])
            self.assertEqual(data["metadata"]["x"], 1)


class TestTelegramText(unittest.TestCase):
    def test_send_text_with_fake_client(self):
        from amber.alerts.telegram import send_telegram_text

        class FakeResp:
            status_code = 200
            text = "ok"

            def json(self):
                return {}

        class FakeClient:
            def __init__(self):
                self.calls = []

            def post(self, url, json):
                self.calls.append((url, json))
                return FakeResp()

        client = FakeClient()
        ok = send_telegram_text("system degraded", bot_token="t", chat_id="c", client=client)
        self.assertTrue(ok)
        self.assertEqual(client.calls[0][1]["text"], "system degraded")


if __name__ == "__main__":
    unittest.main()
