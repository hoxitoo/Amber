# Amber Audit Notes (Debug + Security + ML)

## Fixed issues
1. **Spread filter was ineffective**
   - `bid/ask` were not propagated to feature rows and scanner context.
   - Fixed by adding `bid`, `ask`, `spread_bps` in `FeatureEngine` outputs and passing through scorer.

2. **Dataset quality leakage via synthetic candles**
   - Synthetic rows could be included in training examples.
   - Fixed by excluding rows marked `is_synthetic=True` during dataset row assembly.

3. **Calibration/Eval robustness**
   - Empty dataset produced undefined behavior.
   - Added explicit validation errors for empty dataset in calibrate/eval.

4. **Config path traversal hardening**
   - Config loader now resolves paths and blocks escapes outside project root.

## Remaining high-priority improvements
- Replace scalar calibration with isotonic (production requirement).
- Replace stub model with LightGBM dual-model pump/dump.
- Add walk-forward split with gap enforcement in code, not only docs.
- Add schema validation for dataset/model artifact payloads.
- Add tests (unit + smoke) and CI pipeline.
