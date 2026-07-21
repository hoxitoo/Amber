# Project Amber

Amber — local-first ML scanner for Bybit futures that predicts **event probabilities** (`pump/dump`) and emits alerts.

> Amber is not an auto-trading bot. No order execution logic is in scope.

## Current state (July 20, 2026)
- End-to-end pipeline works on real Bybit v5 contracts: WS collector (kline + tickers) → incremental idempotent normalize → features → dataset → LightGBM train/calibrate/eval → scanner/alerts.
- ML is leakage-safe: time-ordered dataset, walk-forward CV with purge gaps, dedicated calibration holdout and test segments (`in_sample` flag in eval), model-driven backtest on the test split as promotion gate.
- Monitoring is honest: health checks the paths the pipeline actually writes, rolling AUC uses confirmed real outcomes only, PSI compares live features against the train reference stored in the model artifact.
- Alerts: console, Telegram (Bot API) and Discord (webhook) transports; credentials from env vars only. Cooldown/concurrency state persists across runs.
- Deployment profile: systemd units + docker-compose in `deploy/`.

## Quickstart
```bash
bash scripts/setup_env.sh
source .venv/bin/activate

# live ingestion (long-running) + periodic normalize
python scripts/run_ws_collector.py          # WS: kline + tickers -> data/raw/ws_raw
python scripts/run_normalize.py             # incremental, idempotent

# or REST backfill of recent history (shares dedup watermark with WS path)
python scripts/run_collectors.py

python scripts/run_features.py
python scripts/build_dataset.py
python scripts/train_model.py               # LightGBM dual-model + isotonic calibration + eval
python scripts/run_backtest.py              # model-driven replay on the test split
python scripts/run_scanner.py               # one pass; add --loop for the long-running mode
python scripts/run_tests.py
```

## Monitoring
```bash
python scripts/health_check.py    # data/model freshness
python scripts/drift_check.py     # per-feature PSI vs train reference
python scripts/quality_check.py   # confirmed-outcome AUC, bias, PSI
python scripts/report.py          # overall_ok gate for dashboards/uptime checks
```

## Dashboard (UI)
A Streamlit dashboard shows system status, live signals, model quality,
data/drift, per-symbol charts, and a **control panel** to run everything with
buttons (no manual scripts).

**One-click launch** (creates a venv and installs deps on first run):
- Windows: double-click `Amber.bat`
- macOS: double-click `launch.command` (first time: right-click → Open)
- Linux: `./launch.command`

Or manually:
```bash
pip install -r requirements-dashboard.txt
streamlit run amber/dashboard/app.py        # or: python scripts/run_dashboard.py
```
Opens at `http://localhost:8501`. Run it from the project root. It degrades
gracefully before you have data or a trained model.

The **⚙️ Управление** tab lets you: edit the symbol list, start/stop the WS
collector and scanner loop, run each pipeline stage, or run the full
`normalize → features → dataset → train → backtest` cycle — all from the UI.

A standalone executable can be built with PyInstaller — see `packaging/README.md`
(the launcher above is the recommended, reliable path).

## Alerts
Set credentials via environment (never in config/code):
- Telegram: `AMBER_TG_TOKEN`, `AMBER_TG_CHAT`, then add `telegram` to `alerts.channels` in `config/amber.yaml`.
- Discord: `AMBER_DISCORD_WEBHOOK` + `discord` channel.

## Key docs
- `context.md` — product/technical context and invariants.
- `roadmap.md` — stage progress (`[x] / [~] / [ ]`).
- `plan.md` — tactical near-term execution notes.
- `deploy/README.md` — VPS/systemd/docker deployment.
- `codex.md` — session continuity guide.

## Principles
- Event prediction, not directional guessing.
- Offline/live feature parity (no train/serve skew).
- Leakage-safe validation (time splits, purge gaps, censored-row exclusion).
- Calibrated probabilities for decisions.
- Reproducible artifacts and safety-first IO.

## Test runner dependency behavior
- `python scripts/run_tests.py` fails fast when dependencies are missing.
- To auto-install from `requirements.txt` before running tests:
  `AMBER_AUTO_INSTALL_DEPS=1 python scripts/run_tests.py`

## System report readiness semantics
- `python scripts/report.py` returns detailed and summary readiness blocks.
- `overall_ok` reflects `health`, `backtest`, and (optionally) model-eval freshness.
- Model-eval freshness policy is configurable in `config/amber.yaml`
  (`monitoring.model_eval_fresh_sec`, `monitoring.require_model_eval_for_overall_ok`).
- Useful fields: `model_eval_status` (`missing|stale|fresh`), `model_eval_reasons`,
  `overall_reasons`, `readiness`.

## Remaining owner inputs for production go-live
1. Target symbol universe (recommended: top 20–50 liquid USDT-perpetuals) in `config/amber.yaml`.
2. VPS host; deploy per `deploy/README.md`.
3. Telegram bot token / chat id for production alerts (env vars above).
4. Acceptance thresholds for the promotion gate (precision floor, spread/cooldown/concurrency limits) in `config/thresholds.yaml`.
