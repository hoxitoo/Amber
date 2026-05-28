# Project Amber — Context Snapshot (May 22, 2026)

## 1. Product scope
Amber is a real-time ML scanner for Bybit futures. It provides:
- event probabilities (`P(pump)`, `P(dump)`),
- filtering/scoring,
- alerting and monitoring,
- but **no trade execution**.

## 2. What is implemented now
- Contracts/safety: config loader, path guards, manifests, state store, schemas.
- Exchange flow: WS parsing stubs + normalization + gap-fill (`is_synthetic`).
- Feature stack: unified feature engine (offline/live parity).
- Dataset + labeling: adaptive thresholds and synthetic-row exclusion.
- Modeling: baseline dual-head train/infer + calibration + eval + registry/splits.
- Signals: scorer, filters, universe selection, explanations.
- Monitoring: health, drift/PSI/AUC/bias, quality/system reporting.
- Backtest: event-oriented replay metrics.
- CLI apps/scripts and CI tests.

## 3. Current engineering invariants
- No train/serve skew.
- Dataset/model/calibration artifacts are versioned and traceable.
- Synthetic gap-fill rows are not used for model training targets.
- Runtime must degrade safely under partial bad input.

## 4. Current known direction
From roadmap stages:
- Stage 1 mostly complete (foundation).
- Stage 2/3 partially complete (calibration/CV/quality/backtest in progress quality).
- Stage 4/5 largely pending (production ops, API/dashboard, packaging).

## 5. Recent updates (latest sessions)
- `build_system_report` now includes artifact readiness (`artifacts.model_ready`).
- WS normalization made resilient against malformed JSONL lines via streaming parser with warning logs instead of hard failure.
