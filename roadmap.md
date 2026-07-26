# Project Amber — Roadmap

_Last updated: 2026-07-26_

Amber is a local-first ML scanner for Bybit futures that predicts event
probabilities (pump/dump) and emits alerts. Not an auto-trader.

Legend: `[x]` done · `[~]` partial · `[ ]` planned.

---

## Stage 1 — Stable local ML scanner foundation · **done**
- [x] Local pipeline skeleton and runnable scripts.
- [x] Real Bybit v5 WS client (kline + tickers → bid/ask/OI/funding), REST backfill.
- [x] `NormalizedRow` strict validation (Pydantic v2), gap-fill + `is_synthetic`.
- [x] Idempotent ingestion (byte offsets + per-symbol ts watermarks; safe re-runs).
- [x] Unified `FeatureEngine` (offline/live parity, no train/serve skew).
- [x] Signal schema (`SignalV1`), manifests, artifact registry.

## Stage 2 — Modeling correctness and trust · **done**
- [x] LightGBM pump/dump dual-model (logreg fallback, constant-head for single-class).
- [x] Adaptive volatility-based labeling, multi-horizon, censored-row exclusion.
- [x] Time-based walk-forward CV with purge gaps; dedicated calib/test segments.
- [x] Isotonic calibration on a held-out segment; out-of-sample eval (`in_sample` flag).
- [x] SHAP-style feature contributions in explanations.
- [x] Breakout / momentum-precursor feature pack (18 features: volume surge,
      volatility squeeze, breakout geometry, OI rate-of-change).
- [x] Directional score + spread/cooldown/concurrency risk filters (persistent state).

## Stage 3 — Monitoring and validation · **done**
- [x] Health check on the paths the pipeline actually writes.
- [x] Rolling AUC over **confirmed real outcomes** (no self-referential labels).
- [x] Per-feature PSI vs train reference; prediction-bias monitor.
- [x] Model-driven event backtest on the test split (promotion gate).
- [x] PR-AUC + lift-over-base-rate + reliability curve in eval (audit Q7).

## Stage 4 — Production readiness · **done**
- [x] Telegram + Discord transports (env credentials), router warns on unknown channels.
- [x] Universe selection with liquidity floor + warm-up gating.
- [x] systemd units + docker-compose (`deploy/`); mainnet/testnet switch.
- [x] Atomic state writes + single-instance locks on all stages (audit A2/A3).
- [x] Warm-up-row gating in dataset and scanner (audit Q6).

## Stage 5 — Productization · **in progress**
- [x] Streamlit dashboard (status, live signals, model quality, data/drift, per-symbol).
- [x] Control panel: start/stop services, run pipeline stages, symbol editor — all in UI.
- [x] One-click launchers (`Amber.bat`, `launch.command`) + PyInstaller packaging.
- [x] CI: lint + 3.11/3.12 test matrix; Windows build workflow (downloadable package).
- [ ] Model registry rollback workflow from the UI.
- [ ] Parquet storage backend with date partitioning (audit A1).
- [ ] Push observability: alert on `overall_ok` flip / collector death (audit A6).

---

## Current state (honest)

Everything above the line is **built, tested (127 tests), and works end-to-end on
synthetic data**. The machinery is sound. What is **not** yet proven is the one
thing that matters commercially:

> **Does the model have real predictive edge?** All positive metrics to date are
> from synthetic data with signal injected on purpose. This can only be answered
> by collecting real Bybit mainnet data and reading out-of-sample PR-AUC.

See `docs/audit_review_board_2026-07.md` for the full institutional audit.

---

## Future steps — audit-driven backlog

### Sprint 1 — prove-or-kill + cheap correctness
- [x] A2 atomic state writes · A3 single-instance locks · Q6 warm-up gating ·
      Q7 PR-AUC + reliability · Q1 universe logging + `list_instruments`.
- [ ] **Q2 — prove edge on real data (owner action).** Collect ≥2–4 weeks of real
      mainnet data over a rule-based universe; read purged out-of-sample PR-AUC;
      declare a kill criterion up front. *Not a code task — needs real collection.*

### Sprint 2 — statistical validity & robustness · **code done**
- [x] Q3 uniqueness sample weights (1/horizon) for overlapping event windows.
- [x] Q4 per-regime evaluation (trend/range × high/low vol) in verbose eval.
- [x] Q5 label threshold lagged one bar (decoupled from the current bar's ret_1).
- [x] M1 coherent probabilities: joint normalization, p_none exposed in signals.
- [x] M2 class imbalance: scale_pos_weight (LightGBM) / balanced logreg fallback.
- [x] M4 winsorization: train-time clip bounds stored in the model, applied at inference.
- [x] T2 1-bar execution lag in the backtest (decision bar i → entry/outcome bar i+1).
- [x] A4 buffered WS writer (queue + batch flush off the read loop).
- [x] A5 sha256 of all configs embedded in every artifact manifest.
- [x] A6 push alert on overall_ok flip (Telegram text via report run).
- [ ] M3 correlation prune / permutation importance · M5 rolling recalibration cadence
      (deferred: needs real-data volume to be meaningful).

### Sprint 3 — market realism & scale
- [~] T3 order-flow: taker aggressor imbalance / CVD from WS `publicTrade`
      (`taker_imbalance`, `cvd_norm_20`, `trade_count_z_20`) — **done**;
      order-book depth/imbalance (`orderbook`) and liquidations still pending.
- [ ] T1 depth-aware fills + capacity in the backtest (collect L2).
- [ ] T4 wash/manipulated-volume filter · T5 contract-state (ST/delisting) awareness.
- [ ] A1 Parquet partitioning · M6 artifact schema validation.

### Sprint 4 — signal transparency & outcome tracking · **planned (not started)**

Requested 2026-07-26. Make each signal self-explanatory and self-scoring: show
how confident the model is, and later whether it was right.

- [ ] **S4.1 — Per-signal confidence in the UI.** Surface the model's confidence
      for every signal in the "Модель" tab signal list, e.g. `pump — уверенность
      70%`. The number is the head's **calibrated** probability
      (`prob_up_calibrated` / `prob_down_calibrated`), which the model already
      produces — this item is primarily (a) displaying it clearly next to each
      signal and (b) making the calibration trustworthy enough that "70%" really
      means ~70% hit rate (reliability-curve check; recalibrate if the curve
      drifts). Show the direction's own probability, and optionally `p_none`.
      *Model-side note:* confidence is not a new model output — a well-calibrated
      probability IS the confidence. The work is calibration quality + honest
      display, not inventing a separate confidence score.
- [ ] **S4.2 — Realized outcome per signal.** After a signal's horizon elapses,
      resolve whether the predicted event actually happened (pump/dump hit vs
      miss vs timeout) and show the verdict next to the signal (e.g. `dump →
      сработал` / `не сработал` / `таймаут`). Persist the resolution so the
      signal log becomes a track record.
      *Infra that already exists to build on:* the quality monitor already
      lag-joins `signals.jsonl` with realized `normalized` labels after the
      horizon to compute rolling AUC over **confirmed** outcomes — S4.2 reuses
      that join to stamp a per-signal hit/miss and expose it in the UI, plus a
      running hit-rate summary (and, once volume allows, hit-rate broken down by
      confidence bucket to validate S4.1's calibration end-to-end).

### Backlog
- [ ] T6 cross-exchange lead/lag features · A7 feature-list relocation.
- [ ] API + multi-tenant observability (SaaS direction).
