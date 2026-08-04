"""Regression tests for the 2026-07 senior ML audit findings.

B1  dashboard PR-AUC was filtered out of the system report
B2  system-report backtest ignored configured thresholds
B3  absolute 0.65 gate is unreachable for calibrated rare events
B4  constant train feature produced a spurious PSI ~12 "high drift" alarm
B4b discrete/binary features produced the same spurious 12.434 alarm
B5  8-digit hex colour crashed the per-symbol chart
B6  Precision@thr read 0.000 at an unreachable absolute cut
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from amber.monitoring.drift import psi_from_quantile_reference
from amber.models.train import _feature_quantiles
from amber.signals.filters import base_rate_for, effective_prob_min


class TestB3BaseRateGate(unittest.TestCase):
    def test_lift_is_applied_to_odds_not_probability(self):
        """B7: a probability lift breaks at higher base rates — 2x a 0.32 base
        rate demands 0.64, which a calibrated model never reaches, so nothing
        fires. Odds lift stays selective and reachable at any base rate."""
        thr = {"prob_lift_min": 2.0, "prob_abs_floor": 0.12}
        # 2x the odds of 0.10 -> 0.222 odds -> p = 0.1818
        self.assertAlmostEqual(effective_prob_min(thr, 0.10, absolute_key="x"), 0.1818, places=3)
        # at a 0.32 base rate the cut stays reachable instead of jumping to 0.64
        cut = effective_prob_min(thr, 0.32, absolute_key="x")
        self.assertAlmostEqual(cut, 0.4848, places=3)
        self.assertGreater(cut, 0.32)  # still more selective than the base rate
        # 2x the odds of 0.02 is tiny, so the absolute floor wins
        self.assertAlmostEqual(effective_prob_min(thr, 0.02, absolute_key="x"), 0.12)

    def test_cut_stays_below_one_for_every_base_rate(self):
        thr = {"prob_lift_min": 3.0, "prob_abs_floor": 0.0}
        for base in (0.01, 0.1, 0.3, 0.5, 0.8, 0.95):
            cut = effective_prob_min(thr, base, absolute_key="x")
            self.assertGreater(cut, base, f"cut must be selective at base={base}")
            self.assertLess(cut, 1.0, f"cut must stay reachable at base={base}")

    def test_falls_back_to_absolute_cut_without_lift(self):
        thr = {"pump_prob_calibrated_min": 0.65}
        self.assertEqual(effective_prob_min(thr, 0.10, absolute_key="pump_prob_calibrated_min"), 0.65)

    def test_base_rate_read_from_model_head(self):
        model = {"heads": {"pump": {"type": "lightgbm", "label_rate": 0.137}, "dump": {"label_rate": 0.1}}}
        self.assertAlmostEqual(base_rate_for(model, "pump"), 0.137)
        self.assertAlmostEqual(base_rate_for(model, "dump"), 0.1)
        self.assertEqual(base_rate_for(model, "missing"), 0.0)


class TestB4DegeneratePSI(unittest.TestCase):
    def test_constant_reference_grid_yields_zero_psi(self):
        # Old behaviour: all live mass lands in one bin -> PSI ~12 false alarm.
        edges = [0.0] * 11
        self.assertEqual(psi_from_quantile_reference(edges, [1.0, 2.0, 3.0] * 20), 0.0)

    def test_constant_train_feature_is_not_stored_as_reference(self):
        rows = [{"ret_1": i * 0.001, "taker_imbalance": 0.0} for i in range(200)]
        ref = _feature_quantiles(rows)
        self.assertIn("ret_1", ref)  # varying feature kept
        self.assertNotIn("taker_imbalance", ref)  # constant feature skipped


class TestB1B2Reporting(unittest.TestCase):
    def test_pr_auc_metrics_survive_the_report_filter(self):
        from amber.monitoring.reporting import _latest_eval_metrics

        with tempfile.TemporaryDirectory() as td:
            logs = Path(td)
            (logs / "metrics.jsonl").write_text(
                "\n".join(
                    json.dumps({"metric": m, "value": v})
                    for m, v in (
                        ("model_pr_auc_up_cal", 0.31),
                        ("model_pr_auc_up_lift", 1.9),
                        ("model_brier_up_cal", 0.14),
                        ("model_unrelated", 5.0),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            metrics, _age = _latest_eval_metrics(logs)
            self.assertIn("model_pr_auc_up_cal", metrics)
            self.assertIn("model_pr_auc_up_lift", metrics)
            self.assertNotIn("model_unrelated", metrics)

    def test_load_thresholds_walks_up_to_config(self):
        from amber.monitoring.reporting import _load_thresholds

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config").mkdir()
            (root / "config" / "thresholds.yaml").write_text(
                "thresholds:\n  prob_lift_min: 2.0\n  prob_abs_floor: 0.12\n", encoding="utf-8"
            )
            raw = root / "data" / "raw"
            raw.mkdir(parents=True)
            thr = _load_thresholds({"raw_dir": str(raw)})
            self.assertEqual(thr["prob_lift_min"], 2.0)
            self.assertEqual(thr["prob_abs_floor"], 0.12)


class TestB5DashboardSignalMarker(unittest.TestCase):
    """B5: the 'По символу' chart crashed for any symbol that HAD signals because
    the signal marker used an 8-digit RGBA hex ('#00000055'), which plotly
    rejects for scatter.marker.line.color.

    plotly is an optional dashboard dependency (requirements-dashboard.txt), so
    the check that needs it is skipped where it is absent — the core pipeline is
    installed without it. The source scan below needs no dependency and guards
    the regression everywhere.
    """

    @unittest.skipIf(importlib.util.find_spec("plotly") is None, "plotly not installed (optional dashboard dep)")
    def test_signal_marker_color_is_plotly_valid(self):
        import plotly.graph_objects as go

        # Must not raise ValueError (plotly validates on construction).
        go.Scatter(
            x=[1, 2],
            y=[1.0, 2.0],
            mode="markers",
            marker={"size": 12, "symbol": "triangle-up", "color": "#E8A33D",
                    "line": {"width": 1, "color": "rgba(0,0,0,0.33)"}},
            name="сигналы",
        )

    def test_dashboard_uses_no_8_digit_hex_colors(self):
        import re

        src = Path(__file__).resolve().parents[1] / "amber" / "dashboard" / "app.py"
        offenders = re.findall(r"#[0-9A-Fa-f]{8}\b", src.read_text(encoding="utf-8"))
        self.assertEqual(offenders, [], f"plotly rejects 8-digit hex colors: {offenders}")


class TestB4bDiscreteFeaturePSI(unittest.TestCase):
    """PSI reported a permanent 12.434 'high drift'. That number is the exact
    signature of every live value landing in one bin: binary features like
    breakout_up_20 are ~95% zeros, so their quantile edges tie at 0 and the
    uniform 1/n_bins expectation is simply wrong."""

    def _ref(self, ones_every: int = 20, n: int = 2000):
        from amber.models.train import _feature_quantiles

        rows = [{"breakout_up_20": 1.0 if i % ones_every == 0 else 0.0} for i in range(n)]
        return _feature_quantiles(rows)["breakout_up_20"]

    def test_identical_distribution_reports_no_drift(self):
        from amber.monitoring.drift import psi_from_quantile_reference

        live = [1.0 if i % 20 == 0 else 0.0 for i in range(500)]
        self.assertLess(psi_from_quantile_reference(self._ref(), live), 0.1)

    def test_real_drift_is_still_detected(self):
        """The false alarm must not be silenced by making the feature blind."""
        from amber.monitoring.drift import psi_from_quantile_reference

        drifted = [1.0 if i % 10 < 6 else 0.0 for i in range(500)]
        self.assertGreater(psi_from_quantile_reference(self._ref(), drifted), 0.2)

    def test_continuous_features_are_unaffected(self):
        import random

        from amber.models.train import _feature_quantiles
        from amber.monitoring.drift import psi_from_quantile_reference

        rng = random.Random(1)
        ref = _feature_quantiles([{"ret_1": rng.gauss(0, 0.01)} for _ in range(3000)])["ret_1"]
        self.assertLess(psi_from_quantile_reference(ref, [rng.gauss(0, 0.01) for _ in range(1000)]), 0.1)
        self.assertGreater(psi_from_quantile_reference(ref, [rng.gauss(0.03, 0.01) for _ in range(1000)]), 0.2)

    def test_legacy_tied_edges_report_nothing_not_a_fake_alarm(self):
        from amber.monitoring.drift import psi_from_quantile_reference

        legacy = [0.0] * 10 + [1.0]  # old artifact format, edges tied at 0
        self.assertEqual(psi_from_quantile_reference(legacy, [0.0] * 400 + [1.0] * 100), 0.0)


class TestB6ReachableEvalThreshold(unittest.TestCase):
    """Precision@thr read 0.000 because the eval cut was an absolute 0.7 that a
    calibrated rare-event head never reaches — the same mistake as B3, in the
    metric rather than the gate."""

    def test_operating_point_is_reachable(self):
        from amber.signals.filters import effective_prob_min

        thr = {"prob_lift_min": 2.0, "prob_abs_floor": 0.12}
        operating = effective_prob_min(thr, 0.156, absolute_key="pump_prob_calibrated_min")
        self.assertLess(operating, 0.35)  # inside the calibrated range
        self.assertGreater(operating, 0.156)  # still above the base rate

    def test_eval_uses_the_operating_point_when_given(self):
        import inspect

        from amber.models.eval import evaluate_model

        self.assertIn("thresholds", inspect.signature(evaluate_model).parameters)


if __name__ == "__main__":
    unittest.main()
