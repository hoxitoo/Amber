# Project Amber — Implementation Plan (Updated)

## A. What changed from prior plan
1. Scope moved from generic ML scanner to **Bybit futures-specific LightGBM pipeline**.
2. Time architecture explicitly became **Fast/Main/Regime**.
3. Data contract tightened via `NormalizedRow` and gap-fill semantics.
4. Labeling became **adaptive volatility-based** with max-excursion logic.
5. Monitoring scope expanded from infra health to **model quality drift**.
6. Backtest became a required gate before live deployment.

## B. Next execution steps (short horizon)
1. Implement `exchange/schemas.py`, `exchange/normalizer.py`, `exchange/bybit_public.py`.
2. Replace collector stub with WS ingestion over 50+ symbols and persistence to `data/normalized/`.
3. Build `FeatureEngine` class with shared offline/live interface and feature groups v1.
4. Add adaptive labeling in `pipeline/labeling.py` with leakage-safe split gap.
5. Switch model training to LightGBM + isotonic calibration + fold logging.
6. Add SHAP top-3 extraction in infer/scorer.
7. Implement full risk filter chain:
   - prob threshold,
   - directional score,
   - spread cap,
   - per-symbol cooldown,
   - concurrent cap.
8. Implement rolling AUC + PSI monitors and wire alerts.
9. Add event backtester and baseline report.

## C. Acceptance criteria for next milestone
- End-to-end run on real WS data for >=24h.
- Deterministic dataset/model artifacts with manifests.
- Calibrated probabilities used in all signal gating.
- Alerts include top drivers and risk context.
- Monitoring emits health + quality drift warnings.
