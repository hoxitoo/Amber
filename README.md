# Project Amber

HEAD
Amber — local-first ML scanner for Bybit futures that predicts **event probabilities** (`pump/dump`) and emits alerts.

> Amber is not an auto-trading bot. No order execution logic is in scope.

## Current state (May 22, 2026)
- End-to-end prototype exists: collector → normalize → features → dataset → train/calibrate/eval → scanner/alerts.
- Core monitoring and reporting are implemented (health, drift, quality, system report).
- Test suite is active and passing in repository CI/local runner.
- Recent hardening added:
  - resilient WS normalization parsing (invalid JSONL lines are skipped with warnings),
  - report-level artifact readiness (`artifacts.model_ready`).
=======
Local-first ML scanner for Bybit futures focused on **probabilistic pump/dump event detection**.

> Amber is **not** an auto-trading bot. It produces calibrated event probabilities and alerts.

## Current status
Repository contains a runnable local prototype:
- collector -> features -> dataset -> model train/calibration/eval -> scanner,
- manifests + registry + basic monitoring checks,
- console alerts and signal persistence.

## Updated product direction
Based on latest additions, Amber evolves toward:
- Bybit WS real-time ingestion (kline/ticker/OI/funding),
- 3-layer time logic (Fast/Main/Regime),
- LightGBM dual-model (pump/dump),
- adaptive labeling and walk-forward evaluation,
- isotonic calibration,
- SHAP explanations,
- rolling AUC + PSI monitoring,
- event-based backtesting.
>>>>>>> origin/main

## Quickstart
```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/run_collectors.py
python scripts/run_features.py
python scripts/build_dataset.py
python scripts/train_model.py
python scripts/run_scanner.py
<<<<<<< HEAD
python scripts/run_tests.py
```

## Key docs
- `context.md` — product/technical context and invariants.
- `roadmap.md` — stage progress (`[x] / [~] / [ ]`).
- `plan.md` — tactical near-term execution notes.
- `codex.md` — session continuity guide for future Codex runs.

## Principles
- Event prediction, not directional guessing.
- Offline/live feature parity (no train/serve skew).
- Leakage-safe validation.
- Calibrated probabilities for decisions.
- Reproducible artifacts and safety-first IO.
=======
python scripts/health_check.py
python scripts/drift_check.py
```

## Documentation files
- `context.md` — project context and invariants.
- `plan.md` — implementation plan and next execution sequence.
- `roadmap.md` — staged roadmap and completion tracking.

## Core principles
- event prediction, not direction guessing,
- no train/serve skew,
- calibrated probabilities,
- leakage-safe validation,
- explainable signals,
- reproducible artifacts.
>>>>>>> origin/main
