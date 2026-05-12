# Project Amber — Updated Roadmap

## Stage 1 — Stable local ML scanner foundation
- [x] Local pipeline skeleton and runnable scripts.
- [x] Basic collector/features/dataset/model/scanner flow.
- [x] Signal schema contract (`SignalV1`) and manifests.
- [x] Health/metrics/drift initial monitoring primitives.
- [~] Replace stub data ingestion with real Bybit WS/REST client (`kline/ticker/OI/funding`).
- [x] Implement `NormalizedRow` + strict validation (Pydantic v2).
- [~] Add gap-fill and `is_synthetic` handling in IO.
- [x] Introduce unified FeatureEngine (offline/live parity).

## Stage 2 — Modeling correctness and trust
- [ ] Move to LightGBM pump/dump dual-model setup.
- [ ] Adaptive threshold labeling with rolling volatility and multi-horizon support.
- [~] Walk-forward CV with explicit `gap_candles=30` across folds.
- [~] Isotonic calibration on holdout split.
- [ ] SHAP top-features in signal explanation.
- [x] Directional score + spread/cooldown/concurrency risk filters.

## Stage 3 — Monitoring and strategy validation
- [~] Rolling AUC monitor over latest confirmed events.
- [~] PSI monitor per key feature with alert thresholds.
- [~] Prediction bias monitor (`mean P(pump)` vs outcomes).
- [~] Event-based backtest with slippage/commission and TP/SL/timeout outcomes.

## Stage 4 — Production readiness
- [ ] Telegram transport and rate limiting.
- [ ] Better universe selection (top-K + anomalies + liquidity constraints).
- [ ] Runtime supervision/recovery for separate processes.
- [ ] Packaging/deployment profile for VPS/cloud.

## Stage 5 — Productization
- [ ] API and dashboard.
- [ ] model registry rollback workflow.
- [ ] SaaS-facing observability + tenant-safe isolation.
