# Project Amber
main
Amber — local-first ML scanner for Bybit futures that predicts **event probabilities** (`pump/dump`) and emits alerts.

> Amber is not an auto-trading bot. No order execution logic is in scope.

## Current state (May 22, 2026)
- End-to-end prototype exists: collector → normalize → features → dataset → train/calibrate/eval → scanner/alerts.
- Core monitoring and reporting are implemented (health, drift, quality, system report).
- Test suite is active and passing in repository CI/local runner.
- Recent hardening added:
  - resilient WS normalization parsing (invalid JSONL lines are skipped with warnings),
  - report-level artifact readiness (`artifacts.model_ready`).

## Quickstart
```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/run_collectors.py
python scripts/run_features.py
python scripts/build_dataset.py
python scripts/train_model.py
python scripts/run_scanner.py
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
