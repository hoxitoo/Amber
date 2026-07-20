# Amber deployment (VPS)

Two supported modes: systemd (recommended for a single VPS) and docker-compose.

## Process model

| Unit | Role | Schedule |
|---|---|---|
| `amber-ws-collector.service` | live WS ingestion (kline + tickers) into `data/raw/ws_raw` | always-on |
| `amber-normalize.timer` | incremental normalize pass (idempotent, offset-based) | every 60s |
| `amber-scanner.service` | scoring loop, filters, alerts (`--loop`) | always-on |
| `amber-retrain.timer` | features → dataset → train/calibrate/eval → backtest gate | daily 02:30 UTC |

## systemd setup

```bash
sudo useradd -r -m -d /opt/amber amber
sudo git clone <repo> /opt/amber && cd /opt/amber
sudo -u amber bash scripts/setup_env.sh
sudo cp deploy/systemd/* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now amber-ws-collector amber-scanner amber-normalize.timer amber-retrain.timer
```

Note: `amber-normalize.service` runs normalize only; add `run_features.py` to its
ExecStart chain (or extend the retrain unit) once you want features refreshed on
the same cadence as normalization.

## Alert credentials

Never commit tokens. Put them in `/etc/amber/alerts.env` (root-owned, mode 600):

```
AMBER_TG_TOKEN=123456:ABC...
AMBER_TG_CHAT=-1001234567890
# optional
AMBER_DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
```

Then add `telegram` to `alerts.channels` in `config/amber.yaml`.

## mainnet vs testnet

`config/amber.yaml` → `exchange.bybit`:
- mainnet: `ws_url: wss://stream.bybit.com/v5/public/linear`, `testnet: false`
- testnet: `ws_url: wss://stream-testnet.bybit.com/v5/public/linear`, `testnet: true`

The REST backfill client follows the `testnet` flag automatically when
`rest_url` is not set explicitly.

## Health / readiness

```bash
python scripts/health_check.py   # data freshness
python scripts/report.py         # overall_ok gate (health + backtest + eval freshness)
```

Wire `scripts/report.py` into your uptime monitoring; `overall_ok: false` with
`overall_reasons` tells you which component degraded.
