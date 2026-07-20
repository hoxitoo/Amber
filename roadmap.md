# Project Amber — Updated Roadmap

## Stage 1 — Stable local ML scanner foundation
- [x] Local pipeline skeleton and runnable scripts.
- [x] Basic collector/features/dataset/model/scanner flow.
- [x] Signal schema contract (`SignalV1`) and manifests.
- [x] Health/metrics/drift initial monitoring primitives.
- [x] Real Bybit WS/REST client (`kline` + `tickers` incl. OI/funding; REST backfill).
- [x] Implement `NormalizedRow` + strict validation (Pydantic v2).
- [x] Gap-fill and `is_synthetic` handling in IO (excluded from training and signaling).
- [x] Unified FeatureEngine (offline/live parity).
- [x] Idempotent ingestion (offsets + watermarks; safe re-runs).

## Stage 2 — Modeling correctness and trust
- [x] LightGBM pump/dump dual-model setup (logreg fallback).
- [x] Adaptive threshold labeling with rolling volatility and multi-horizon support.
- [x] Walk-forward CV with purge gap in time (candles), not row indices.
- [x] Isotonic calibration on a dedicated holdout split.
- [x] SHAP top-features in signal explanation (LightGBM `pred_contrib`).
- [x] Directional score + spread/cooldown/concurrency risk filters (persistent).

## Stage 3 — Monitoring and strategy validation
- [x] Rolling AUC monitor over latest **confirmed** events (real outcome join).
- [x] PSI monitor per key feature vs train reference with alert thresholds.
- [x] Prediction bias monitor (`mean P(pump)` vs `mean P(dump)`).
- [x] Event-based backtest with slippage/commission, model-driven on the test split.
- [ ] 24h+ soak on live mainnet data (needs deployment; see below).

## Stage 4 — Production readiness
- [x] Telegram transport and rate limiting (env-based credentials).
- [x] Better universe selection (top-K + liquidity floor via `notional_volume_1m`).
- [x] Runtime supervision profile (systemd units / docker-compose in `deploy/`).
- [x] Packaging/deployment docs for VPS.
- [ ] Live go-live: owner provides VPS, symbol universe, Telegram credentials,
      and acceptance thresholds; then run the 24h soak and promotion gate.

## Stage 5 — Productization (next iteration)
- [ ] API and dashboard.
- [ ] Model registry rollback workflow.
- [ ] SaaS-facing observability + tenant-safe isolation.
- [ ] Parquet storage backend with date partitioning.
