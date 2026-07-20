# Project Amber — Context Snapshot (July 20, 2026)

## 1. Product scope
Amber is a real-time ML scanner for Bybit futures. It provides:
- event probabilities (`P(pump)`, `P(dump)`),
- filtering/scoring,
- alerting and monitoring,
- but **no trade execution**.

## 2. What is implemented now
- Contracts/safety: config loader, path guards, manifests, state store, schemas.
- Exchange flow: real Bybit v5 WS parsing (symbol from topic, confirm-only,
  tickers snapshot/delta merge for bid/ask/OI/funding), REST backfill client,
  gap-fill with `is_synthetic`, idempotent incremental normalization
  (byte offsets + per-symbol ts watermarks).
- Feature stack: unified feature engine (offline/live parity), full-history
  feature output, `is_synthetic` and `notional_volume_1m` propagated.
- Dataset + labeling: adaptive thresholds, multi-horizon, censored-row
  exclusion, global chronological ordering.
- Modeling: LightGBM dual-head (logreg fallback) + isotonic calibration on a
  dedicated holdout + out-of-sample eval + registry; model artifact stores
  split boundaries, train feature quantiles (PSI reference) and labeling
  metadata.
- Signals: scorer (full feature set, SHAP contributions), persistent
  gate/rate-limiter, universe selection with liquidity floor, explanations.
- Monitoring: health (real paths), confirmed-outcome rolling AUC, per-feature
  PSI vs train reference, bias, system report with readiness gates.
- Backtest: model-driven replay through the threshold chain on the test split.
- Alerts: console + Telegram + Discord transports (env credentials).
- Deployment: systemd units + docker-compose (`deploy/`), CI with lint and a
  3.11/3.12 test matrix.

## 3. Current engineering invariants
- No train/serve skew (single FeatureEngine).
- Dataset/model/calibration artifacts are versioned and traceable; feature spec
  version v2 synchronized between code and configs.
- Synthetic gap-fill rows never become training targets or signals.
- Ingestion is idempotent: re-running any stage never duplicates rows.
- Train/calib/test segments are time-separated with purge gaps; eval carries an
  explicit `in_sample` flag.
- Runtime degrades safely under partial bad input (tolerant JSONL everywhere).
- Secrets only via environment variables.

## 4. Current direction
Stages 1–3 complete except the live 24h soak; Stage 4 code-complete, pending
owner go-live inputs. Stage 5 (API/dashboard, parquet storage) is the next
iteration.

## 5. Owner inputs needed for go-live
1. Symbol universe (recommend top 20–50 liquid USDT-perps) in `config/amber.yaml`.
2. VPS/host for `deploy/` profile.
3. Telegram credentials (`AMBER_TG_TOKEN`/`AMBER_TG_CHAT`).
4. Acceptance thresholds for the promotion gate (`config/thresholds.yaml`).
