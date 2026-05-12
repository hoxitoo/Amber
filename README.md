# Project Amber

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

## Quickstart
```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/run_collectors.py
python scripts/run_features.py
python scripts/build_dataset.py
python scripts/train_model.py
python scripts/run_scanner.py
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
