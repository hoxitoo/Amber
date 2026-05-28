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


## Test runner dependency behavior
- `python scripts/run_tests.py` now fails fast when dependencies are missing.
- To auto-install from `requirements.txt` before running tests, use:
  `AMBER_AUTO_INSTALL_DEPS=1 python scripts/run_tests.py`

## System report readiness semantics
- `python scripts/report.py` returns both detailed and summary readiness blocks.
- `overall_ok` now reflects `health`, `backtest`, and (optionally) model-eval freshness.
- Model-eval freshness policy is configurable in `config/amber.yaml`:
  - `monitoring.model_eval_fresh_sec` — freshness window in seconds.
  - `monitoring.require_model_eval_for_overall_ok` — whether stale/missing eval metrics should fail `overall_ok`.
- Useful report fields:
  - `model_eval_status`: `missing | stale | fresh`
  - `model_eval_reasons` / `overall_reasons` (structured diagnostics)
  - `readiness` (compact status snapshot for dashboards)

## External blockers (required from project owner)
Without the items below, development can continue only in local/synthetic mode and cannot fully close Stage 1/2 production criteria:

1. **Bybit runtime data access (Stage 1 blocker)**
   - Confirm `mainnet` vs `testnet`.
   - Provide target symbol universe (ideally 20–50+).
   - Confirm required streams (`kline`, `ticker`, `open interest`, `funding`).
2. **Production runtime target**
   - VPS/cloud host details (CPU/RAM/disk), process model (`systemd`/docker), and storage path policy.
3. **Alert transport credentials**
   - Telegram bot token and destination chat/channel IDs (when enabling prod alerts).
4. **Product/risk acceptance thresholds**
   - Required precision floors, spread/cooldown/concurrency limits, and model promotion gates.
