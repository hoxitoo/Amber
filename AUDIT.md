# Amber Audit Notes (Debug + Security + ML)

## Full audit + hardening pass (2026-07-20)

A code audit found the live path effectively non-functional and several
"always-green"/"always-red" monitors, all masked by tests whose fixtures encoded
the same mistakes. All items below are fixed and covered by tests.

### Data pipeline (was broken on real data)
1. **WS kline parser read `item["symbol"]`** — Bybit v5 sends the symbol only in
   the topic, so every real candle was silently dropped (bare `except` swallowed
   the KeyError). Fixed: symbol parsed from topic, parse errors logged.
2. **`confirm` flag ignored** — in-progress candle updates created duplicate rows
   per minute. Fixed: confirm-only emission + per-symbol ts watermark dedup.
3. **Only the last feature row was written per run** — the dataset was degenerate.
   Fixed: full history written incrementally with a watermark.
4. **Live path had no ticker/OI/funding** — `spread_bps`, `oi_z_20`,
   `funding_z_20` were identically zero. Fixed: WS collector subscribes to
   `tickers.*`; normalizer merges snapshot/delta into the market-state cache.
5. **Append-only sink with no idempotency** — re-running normalize duplicated all
   data. Fixed: byte-offset + watermark state; REST backfill shares the same
   watermark.

### ML correctness
6. **Model was a stub with hardcoded ±8.0 weights** and bias `-(rate-0.5)`.
   Fixed: real LightGBM dual-model (logreg fallback), trained on the train
   segment only.
7. **Symbol-major dataset + row-index CV** leaked across symbols/time. Fixed:
   global chronological ordering, time-based walk-forward with purge gaps.
8. **Train/calibrate/eval all used the same rows** (in-sample metrics). Fixed:
   dedicated calib/test segments stored in the model artifact; eval reports an
   explicit `in_sample` flag.
9. **Backtest ignored the model** and booked every labeled event ×3 horizons.
   Fixed: model+thresholds replay on the test split, one horizon per trade,
   per-symbol open-trade cooldown.
10. **Right-censored rows labeled "no event"**. Fixed: dropped per horizon.
11. **`is_synthetic` filter never fired** (feature rows didn't carry the flag —
    previously claimed fixed here, which was wrong). Actually fixed now:
    the flag propagates through the feature engine and the scanner also skips
    synthetic rows.

### Monitoring
12. **Health checked `raw/ticks/`** which nothing ever wrote → permanently red.
    Fixed: checks `raw/normalized/`.
13. **Rolling AUC used a pseudo-label derived from the prediction itself** →
    permanently ~1.0. Fixed: AUC only over confirmed real outcomes (signal ×
    candles join after the horizon elapses); None when no data.
14. **Drift used a mean-shift test with a 0.05 threshold** (returns are ~1e-3 →
    could never fire) and PSI compared the live stream to itself. Fixed:
    per-feature PSI against train-set quantiles stored in the model artifact.

### Alerts / runtime
15. **Telegram/Discord were uncalled no-op stubs; unknown channels dropped
    silently.** Fixed: real transports (httpx, env credentials, retry/429),
    router warns on unknown channels.
16. **Cooldown/concurrency state lived in memory of a one-shot process**, and
    the concurrency cap never released slots. Fixed: state persists via
    StateStore, slots expire after a TTL.

## Remaining known limitations
- Storage remains JSONL under a `ParquetSink` name; fine at current scale,
  swap to parquet partitioning when volume demands it.
- Timeout trades in the backtest approximate exit PnL as `-cost` (final price
  at horizon end is not stored in dataset rows).
- The system report's backtest falls back to label replay when no model exists;
  the field `mode` distinguishes the two.
