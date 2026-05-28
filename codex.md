# CODEX Session Continuity Guide — Amber

_Last updated: 2026-05-28_

## 1) Purpose of this file
This file preserves operational context for new Codex sessions so development continuity is not lost.

---

## 2) Project summary
Amber is a local-first ML scanner for Bybit futures:
- input: normalized market stream/features,
- output: calibrated `pump/dump` event probabilities + alerts,
- non-goal: trade execution.

Primary flow:
1. Collect raw data
2. Normalize + gap-fill
3. Compute features
4. Build dataset/labels
5. Train + calibrate + evaluate
6. Run scanner + filters + alerts
7. Monitor/report/backtest quality

---

## 3) Current stage and status
Reference: `roadmap.md`.

High-level status:
- **Stage 1 (foundation)**: mostly done.
- **Stage 2 (model correctness)**: partial.
- **Stage 3 (monitoring/validation)**: partial.
- **Stage 4/5 (prod/productization)**: mostly pending.

Recently completed hardening:
- reporting artifact readiness flag (`artifacts.model_ready`),
- WS normalization resilient JSONL parsing (skip bad lines + warning logs).

Current external blockers (owner input required):
- Real Bybit operation profile (`mainnet/testnet`, symbol list, stream scope).
- Deployment target and runtime model (host/resources/supervision/storage).
- Alert transport credentials (Telegram bot/chat config).
- Product acceptance thresholds for promotion/risk gates.

---

## 4) Development workflow for Codex
When continuing work, follow this order:

1. **Read context first**
   - `README.md`, `context.md`, `roadmap.md`, `plan.md`, this file.
2. **Run baseline tests**
   - `python scripts/run_tests.py`.
3. **Choose one bounded improvement**
   - bugfix, security hardening, or roadmap item slice.
4. **Implement incrementally**
   - avoid broad rewrites; preserve existing structure.
5. **Add/adjust tests for changed behavior**
   - prefer targeted unit tests.
6. **Run full tests again**
   - ensure no regressions.
7. **Commit and update continuity docs if needed**
   - update this file when stage/architecture assumptions shift.

---

## 5) Security/debug checklist for future sessions
Before finalizing changes, check:

- Input parsing robustness (malformed JSONL should not crash pipeline unless explicitly required).
- Path traversal protections for filesystem writes/reads.
- Secret handling (no hardcoded tokens/keys in code or logs).
- Unbounded memory reads on large files (prefer streaming when possible).
- Error handling with actionable logs (file, line, context).
- Tests covering failure-path branches.

---

## 6) Key commands
```bash
# tests
python scripts/run_tests.py

# common pipeline
python scripts/run_collectors.py
python scripts/run_normalize.py
python scripts/run_features.py
python scripts/build_dataset.py
python scripts/train_model.py
python scripts/run_scanner.py

# monitoring
python scripts/health_check.py
python scripts/drift_check.py
python scripts/quality_check.py
python scripts/report.py
```

---

## 7) Repo structure quick map
- `amber/common` — config/logging/paths/manifests/types
- `amber/exchange` — schemas, normalizer, streams/stubs
- `amber/features` — online/offline feature logic
- `amber/datasets` — dataset assembly/labeling
- `amber/models` — train/infer/eval/calibration/registry
- `amber/signals` — score/filter/explain/universe
- `amber/monitoring` — health/drift/quality/reporting
- `amber/backtest` — event backtest
- `amber/pipeline` — runnable app entrypoints
- `scripts/` — CLI wrappers and utilities
- `tests/` — unit tests

---

## 8) Notes for next Codex
- Keep architecture stable and evolve by small safe steps.
- Prioritize roadmap items marked `[~]` before large new feature branches.
- Any change in invariants must be reflected in `context.md` + `codex.md`.
- If owner-provided blockers are unresolved, prefer internal robustness/observability work and document the dependency explicitly in session output.
