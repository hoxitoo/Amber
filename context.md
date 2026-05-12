# Project Amber — Updated Context (May 2026)

## 1) Product Positioning
Project Amber is a **real-time ML market scanner** for Bybit futures focused on detecting probable short-term pump/dump events.

Amber is **not** an auto-trading system:
- no order execution,
- no private API keys required for core scanner,
- alerts + decision support only.

## 2) Core prediction target
Amber predicts **events** (not candle direction):
- `P(pump)` — probability of reaching an upward move threshold within horizon `H`,
- `P(dump)` — probability of reaching a downward move threshold within horizon `H`.

## 3) Key updates vs previous context
New additions integrated into product context:

1. **LightGBM-first strategy** (instead of generic baseline models).
2. **Multi-layer time logic**:
   - Fast: 15s (anomaly trigger),
   - Main: 1m (main inference),
   - Regime: 15m (market context gate).
3. **NormalizedRow contract** with required fields:
   `ts, symbol, tf, ohlc, volume, bid, ask, oi, funding, is_synthetic`.
4. **Gap-fill policy** for missing candles with `is_synthetic=True`, excluded from training.
5. **Feature scope expansion** to 60+ features across volume/OI/funding/spread/SR/cross-asset/microstructure.
6. **Adaptive labeling threshold**: `k * rolling_volatility`, bounded `[0.3%, 5.0%]`.
7. **Max-excursion labeling** preferred over close-to-close.
8. **Walk-forward only** with train/val gap (`gap_candles=30`).
9. **Calibration expectation** upgraded to isotonic (preferred production target).
10. **Directional score gate** required for signal quality.
11. **Quality monitoring** requires Rolling AUC + PSI + prediction bias checks.
12. **Event-based backtest** added as required pre-live stage.

## 4) Architecture direction (target)
- `common`: config, logging, io, schema contracts
- `exchange`: WS/REST + normalization
- `pipeline/ta`: feature engine, labeling, S/R
- `models`: train/infer/registry
- `signals`: scorer + alert formatter/transports
- `monitoring`: drift + online quality monitors
- `backtest`: event replay and PnL diagnostics

## 5) Engineering invariants
- No train/serve skew (single FeatureEngine logic for offline/live).
- Strict artifact versioning (dataset/model/calibration/signal schema refs).
- Calibrated probabilities only for threshold decisions.
- Regime-aware, leakage-safe evaluation.
- Observable runtime (lag, drift, quality, alert volume).
