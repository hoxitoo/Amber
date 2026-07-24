# Amber deployment (VPS)

Recommended: systemd with three long-running services that mirror the dashboard's
three processes. docker-compose is an alternative.

## Location matters (Bybit access)

Bybit geo-restricts some regions. Deploy in the **EU** (Netherlands, Germany,
Finland) where the public API/WS is reachable. Avoid US and Russian IPs — the WS
stream may fail to connect there.

## Process model

| Unit | Role |
|---|---|
| `amber-ws-collector.service` | live WS ingestion (kline + tickers + publicTrade) into `data/raw/ws_raw` |
| `amber-pipeline.service` | loop: normalize + features every ~60s, rebuild dataset + retrain every `pipeline.retrain_min` |
| `amber-scanner.service` | scoring loop, filters, alerts (`--loop`) |
| `amber-dashboard.service` | optional Streamlit UI on `127.0.0.1:8501` (view over an SSH tunnel) |

These are the same three (+UI) processes the dashboard's control panel starts;
on a VPS systemd supervises them instead.

## systemd setup (Ubuntu 22.04)

```bash
# as root
apt update && apt install -y python3-venv python3-pip git
useradd -r -m -d /opt/amber amber
git clone https://github.com/hoxitoo/amber /opt/amber
chown -R amber:amber /opt/amber
cd /opt/amber
sudo -u amber bash scripts/setup_env.sh
sudo -u amber .venv/bin/pip install -r requirements-dashboard.txt   # for the UI

cp deploy/systemd/amber-*.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now amber-ws-collector amber-pipeline amber-scanner amber-dashboard
```

Edit `config/amber.yaml` first (symbols, `storage.keep_runs`, `collect_trades`)
so the collected universe and disk retention match the box.

## View the dashboard (SSH tunnel)

The UI binds to localhost only. From your PC:

```bash
ssh -L 8501:127.0.0.1:8501 amber@<server-ip>
```

Then open `http://localhost:8501` in your browser.

## Alert credentials

Never commit tokens. Put them in `/etc/amber/alerts.env` (root-owned, mode 600)
and reference it from the scanner unit (`EnvironmentFile=-/etc/amber/alerts.env`):

```
AMBER_TG_TOKEN=123456:ABC...
AMBER_TG_CHAT=-1001234567890
# optional
AMBER_DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
```

Then add `telegram` to `alerts.channels` in `config/amber.yaml`.

## Disk

`ws_raw` self-rotates and is cleaned by the normalizer; datasets/models are
pruned to `storage.keep_runs`. To further cap disk: fewer symbols, or
`exchange.bybit.collect_trades: false` (drops order-flow features). Run
`python scripts/cleanup.py` for an on-demand sweep.

## Health / readiness

```bash
.venv/bin/python scripts/health_check.py   # data freshness
.venv/bin/python scripts/report.py         # overall_ok gate
```

## Logs

```bash
journalctl -u amber-ws-collector -f
journalctl -u amber-pipeline -f
journalctl -u amber-scanner -f
```

## Alternative: docker-compose

`docker compose -f deploy/docker-compose.yml up -d` (see that file). The systemd
path above is simpler for a single box.
