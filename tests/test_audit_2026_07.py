"""Regression tests for the 2026-07 senior ML audit findings (B1-B4).

B1  dashboard PR-AUC was filtered out of the system report
B2  system-report backtest ignored configured thresholds
B3  absolute 0.65 gate is unreachable for calibrated rare events
B4  constant train feature produced a spurious PSI ~12 "high drift" alarm
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
    def test_lift_mode_uses_base_rate_when_configured(self):
        thr = {"prob_lift_min": 2.0, "prob_abs_floor": 0.12}
        # 2x a 0.10 base rate = 0.20, above the floor
        self.assertAlmostEqual(effective_prob_min(thr, 0.10, absolute_key="pump_prob_calibrated_min"), 0.20)
        # 2x a 0.02 base rate = 0.04, so the absolute floor 0.12 wins
        self.assertAlmostEqual(effective_prob_min(thr, 0.02, absolute_key="pump_prob_calibrated_min"), 0.12)

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


if __name__ == "__main__":
    unittest.main()
