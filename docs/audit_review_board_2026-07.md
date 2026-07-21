# Project Amber — Independent Review-Board Audit

_Date: 2026-07-21 · Scrutiny level: technical due diligence for a quantitative fund · Verdict below is deliberately critical, not congratulatory._

> **Remediation status (2026-07-21).** Sprint-1 *code* items are implemented:
> A2 atomic StateStore writes (temp + `os.replace`), A3 single-instance file
> lock on all pipeline stages, Q6 warm-up gating (`min_warmup_bars`) in dataset
> and scanner, Q7 PR-AUC + reliability curve in eval (surfaced in the dashboard),
> Q1 universe-membership logging + a REST `list_instruments` for rule-based /
> survivorship-aware selection. **Q2 (prove edge on real data) is not a code
> task and remains open** — it requires real mainnet collection.

## Executive verdict

The **engineering foundation is solid-to-good** after recent hardening (single offline/online feature engine, time-purged splits, out-of-sample eval flag, model-driven backtest, honest confirmed-outcome AUC, manifests/registry, tolerant IO, path guards). The commercial risk does **not** live in the code quality — it lives in two places the board rates **Critical**:

1. **No real statistical edge has been demonstrated.** Every positive number to date is from synthetic data with signal deliberately injected. Whether the features predict *real* pumps is unproven and is the entire thesis.
2. **The backtest has no execution or liquidity realism.** A fixed 5 bps slippage on exactly the thin, low-cap coins that pump is fantasy; backtested edge will not survive live fills.

Everything else is subordinate to resolving these two. Scaling or polishing an unproven-edge system is premature.

**Severity tally:** Critical 3 · High 8 · Medium 8 · Low 3.

---

## Strengths (brief, for balance)

- Single `FeatureEngine` for batch and stream → genuinely no train/serve skew.
- Time-ordered dataset, walk-forward CV with purge gap in candle-time, dedicated calib/test segments, `in_sample` flag in eval, censored-row exclusion.
- Model-driven backtest on the test split (not label replay); confirmed-outcome rolling AUC (no self-referential pseudo-labels).
- Path-traversal guards, tolerant JSONL everywhere, artifact manifests + registry, idempotent ingestion via offsets/watermarks.
- Reasonable unit coverage (116 tests) that asserts real numeric behavior.

---

## 1 · Lead Quant Researcher

### Q1 — Survivorship & selection bias in the universe · **Critical** · Sprint 1
**Description.** Amber trains only on a hand-picked list of *currently listed, currently liquid* symbols. Delisted, dead, and post-pump-collapsed coins never enter the dataset; the user chooses the symbols by hand.
**Why it matters.** Pump/dump events concentrate precisely in low-cap, newly-listed, and about-to-be-delisted names — the population most likely to be absent or hand-filtered out. The training distribution is not the live opportunity distribution.
**Potential impact.** Inflated, non-representative base rates; a model that looks calibrated in-sample and misfires on the real event population; false confidence at go-live.
**Recommended solution.** Snapshot the *full* perpetual universe over time (including symbols later delisted); record listing/delisting dates; sample symbols by rule (liquidity/age bands), not by hand. At minimum, log the universe membership per day so selection is reproducible and auditable.

### Q2 — Statistical edge is unproven; all metrics are synthetic · **Critical** · Sprint 1
**Description.** The only positive results (AUC ≈ 0.64) come from synthetic data with volume/price surges injected on purpose. No real-data out-of-sample result exists.
**Why it matters.** The core hypothesis — that 1m price/volume/OI/funding precursors predict real pumps — is untested. Most such projects die here.
**Potential impact.** The entire product may have no edge; every downstream investment (UI, deploy, features) is at risk until this is answered.
**Recommended solution.** Treat Sprint 1 as prove-or-kill: collect ≥2–4 weeks of real mainnet data across a rule-based universe, then read PR-AUC / precision@k and calibration on a purged out-of-sample window. Define a kill criterion up front (e.g. test PR-AUC not materially above base rate ⇒ features insufficient).

### Q3 — Overlapping-event label autocorrelation · **High** · Sprint 2
**Description.** Adjacent bars and multiple horizons (5/10/20) from the same bar produce heavily overlapping, correlated label windows. Time-purged folds prevent train/val *crossing* leakage, but within a fold the samples are near-duplicates.
**Why it matters.** Effective sample size is far below row count; CV metrics and significance are optimistic; the model can memorize repeated windows.
**Potential impact.** Overstated confidence, poor live generalization.
**Recommended solution.** Add an embargo between train and val equal to the max horizon; de-overlap training samples (sample every H bars) or apply uniqueness/return-attribution sample weights (López de Prado). Report metrics on non-overlapping events.

### Q4 — No regime-conditioned validation · **High** · Sprint 2
**Description.** `regime` is hardcoded `"unknown"`; performance is never stratified by trend/range/high-vol/low-vol.
**Why it matters.** Edge in crypto is regime-dependent; a signal that works in high-vol trending markets can be net-negative in chop.
**Potential impact.** A blended metric hides that the strategy loses money in the dominant live regime.
**Recommended solution.** Compute a regime label (e.g. realized-vol tercile × trend sign over a rolling window) and report precision / PR-AUC / Brier / calibration *per regime*. Gate promotion on per-regime performance, not the blend.

### Q5 — Label threshold coupled to a feature · **High** · Sprint 2
**Description.** The adaptive event threshold = f(rolling vol of `ret_1`), and that same volatility is also a model input.
**Why it matters.** The target is partly *defined* by an input, so the model can exploit the definitional relationship instead of learning real precursors — a subtle circularity.
**Potential impact.** Overstated apparent skill; degradation when the vol/threshold relationship shifts.
**Recommended solution.** Either fix thresholds per liquidity/vol bucket (documented, not magic), or hold the threshold-defining volatility out of the feature set, and verify skill survives.

### Q6 — Warm-up contamination · **Medium** · Sprint 2
**Description.** Rows with `obs` below the longest lookback (~60) emit 0.0 for `ret_60`, rolling z-scores, breakout distances, etc. These degenerate rows are included as training samples and can be scored live.
**Why it matters.** Zeros are not "neutral"; they are wrong values the model learns from and acts on before an instrument has enough history.
**Recommended solution.** Gate a row as train/live-eligible only when `obs ≥ max_lookback`; drop the warm-up prefix per symbol.

### Q7 — ROC-AUC as headline for a rare-event problem · **Medium** · Sprint 1
**Description.** Eval reports ROC-AUC + precision@threshold + Brier; no PR-AUC / average precision, no reliability curve.
**Why it matters.** Under class imbalance, ROC-AUC is optimistic and uninformative about the operating point that matters (high-precision alerts).
**Recommended solution.** Add PR-AUC / average precision, precision@k, and a calibration (reliability) curve; make PR-AUC the headline.

---

## 2 · Professional Crypto Derivatives Trader

### T1 — Backtest has no execution or liquidity realism · **Critical** · Sprint 2
**Description.** Fixed 5 bps slippage + 4 bps fee, filled at the signal bar's price, no depth, no impact, no partial fills.
**Why it matters.** The coins that pump are the thinnest; real slippage and impact on a 15% mover are orders of magnitude larger and asymmetric. Entry into a breakout eats the book.
**Potential impact.** Backtested profit factor / Sharpe are meaningless for sizing; a "profitable" strategy can be net-negative after real fills.
**Recommended solution.** Model fills against order-book depth (collect L2), cap size to a fraction of top-of-book / recent volume, use spread- and volatility-scaled slippage, and stress-test with pessimistic fills. Report capacity (max $ per signal) alongside returns.

### T2 — 1-bar execution lag not modeled · **High** · Sprint 2
**Description.** Features are known at the *close* of bar i; live you can only act at the open of bar i+1, but the backtest enters at bar i's price.
**Why it matters.** For fast breakouts the first post-signal bar is where most of the move happens; entering "at signal" is look-ahead-lite.
**Recommended solution.** Enter backtest trades at bar i+1 open (or VWAP of i+1), and re-measure.

### T3 — Order-flow / liquidation / depth blind spot · **High** · Sprint 3
**Description.** Only 1m OHLCV + ticker (bid/ask/OI/funding). No aggressor-side trade flow / CVD, no liquidation feed, no book imbalance.
**Why it matters.** These are the actual *leading* indicators of ignition; price/OI/funding are comparatively lagging. This is likely the difference between edge and no edge (ties to Q2).
**Recommended solution.** Phase 2: subscribe to `publicTrade` (aggressor imbalance, trade-size distribution, CVD) and `orderbook` (depth imbalance, book pressure); add Bybit liquidation stream. Prioritize if Sprint-1 edge is marginal.

### T4 — Wash / manipulated volume not filtered · **Medium** · Sprint 3
**Description.** `vol_ratio`/`vol_accel` surges can be fake prints on low-caps.
**Why it matters.** Training on manipulated volume teaches the model a signal that vanishes or reverses when the manipulator stops.
**Recommended solution.** Cross-check volume against trade count / unique-size dispersion; down-weight venues/symbols with implausible volume-to-trade ratios.

### T5 — Contract-state blindness (ST / pre-delisting / settlement) · **Medium** · Sprint 3
**Description.** Screenshots show `ST` tags and delisting notices; these states dominate price behavior and are not captured.
**Why it matters.** A model blind to "this contract is being delisted next week" will mis-predict its violent, non-repeatable moves.
**Recommended solution.** Ingest instrument status / announcements; add a contract-state feature or exclude ST/pre-delist symbols from signals.

### T6 — Single-venue view · **Low** · Backlog
**Description.** No cross-exchange context.
**Why it matters.** Pumps frequently originate on spot / Binance and propagate to Bybit perps with a lag — a usable lead signal is left on the table.
**Recommended solution.** Add a reference feed (e.g. Binance spot/perp) for lead/lag features once single-venue edge is confirmed.

---

## 3 · Senior ML / Data Scientist

### M1 — Incoherent probability semantics · **High** · Sprint 2
**Description.** Pump and dump heads are calibrated independently; `p_up + p_down` can exceed 1. The product markets "probability estimation."
**Why it matters.** The numbers aren't a valid distribution over {pump, dump, none}; thresholding them as if they were is unsound and hurts explainability/trust.
**Recommended solution.** Model it as a 3-class problem (pump/dump/none) with a single calibrated multinomial, or calibrate jointly and normalize; document the exact semantics.

### M2 — No class-imbalance handling · **High** · Sprint 2
**Description.** LightGBM trains with default params — no `scale_pos_weight` / `is_unbalance`, no focal objective.
**Why it matters.** Rare-event heads underfit the positive class; calibration and high-precision operating points suffer.
**Recommended solution.** Set class weights from base rates, tune `min_data_in_leaf` / `num_leaves` per data size, and validate that calibration improves; consider a precision-oriented objective.

### M3 — Feature redundancy / multicollinearity · **Medium** · Sprint 2
**Description.** `vol_z_20`/`vol_ratio_20`/`vol_accel`, `ret_1/5/20/60`, `dist_to_high`/`breakout_up` are strongly correlated.
**Why it matters.** Trees tolerate it, but feature importance becomes unstable and PSI/drift attribution and explanations become unreliable.
**Recommended solution.** Track a correlation matrix; prune or combine; report importance with permutation (not split-count); keep the set lean.

### M4 — No winsorization / clipping of unbounded features · **Medium** · Sprint 2
**Description.** `vol_ratio`, `oi_roc`, `vol_accel` are unbounded and spike.
**Why it matters.** Extreme values destabilize the logreg fallback and distort PSI bin edges / reference quantiles.
**Recommended solution.** Winsorize (e.g. 1st/99th pct) or log-transform ratio features; store the clip bounds in the model artifact.

### M5 — Static single-slice calibration; no recalibration cadence · **Medium** · Sprint 2
**Description.** Isotonic on one holdout slice with potentially few positives; no scheduled recalibration, no action tied to calibration decay.
**Why it matters.** Isotonic overfits with few positives; calibration drifts with regime and decays silently.
**Recommended solution.** Prefer Platt/beta calibration when positives are scarce; recalibrate on a rolling window; wire the existing PSI/decay monitor to trigger recalibration.

### M6 — Downstream artifacts are unvalidated dicts · **Low** · Sprint 3
**Description.** `NormalizedRow` is pydantic-validated, but feature and dataset rows are raw dicts; missing columns silently become 0.0.
**Why it matters.** Schema drift erodes correctness invisibly.
**Recommended solution.** Define pydantic schemas (or a column contract + assertion) for feature and dataset rows; fail loudly on missing/renamed columns.

---

## 4 · Senior Software Architect

### A1 — "ParquetSink" writes JSONL; no partitioning or columnar format · **High** · Sprint 3
**Description.** Despite the name, storage is a single append-only `part-000.jsonl` per symbol; every stage reads the whole file.
**Why it matters.** Feature recompute replays *all* history each run (O(N) per run, O(N²) cumulative); no date partitioning, compaction, or predicate pushdown. Fine at MB scale, wall at GB.
**Potential impact.** Ingestion/feature latency grows unbounded; the "commercial platform" framing can't scale on this.
**Recommended solution.** Move to real Parquet partitioned by `symbol/date`; make the feature stage incremental (only new partitions); rename the abstraction to match reality. Defer until Sprint-1 edge justifies the volume.

### A2 — Non-atomic StateStore writes · **High** · Sprint 1
**Description.** `StateStore.set` truncates then writes the file in place; a crash mid-write corrupts watermarks/offsets.
**Why it matters.** These files gate idempotency (what's been normalized). Corruption ⇒ silent duplication or data loss on restart.
**Potential impact.** Data-integrity failure that is hard to detect after the fact.
**Recommended solution.** Write to a temp file + `os.replace` (atomic rename); optionally keep one backup generation. Cheap, do it now.

### A3 — No single-instance guard / file lock · **Medium** · Sprint 2
**Description.** Overlapping normalize runs (timer fires again before the previous finishes) or concurrent collector+normalize can race on offsets.
**Why it matters.** Double-processing / interleaved writes corrupt the normalized series.
**Recommended solution.** A PID/file lock per stage (or a `flock`); make timer units `Type=oneshot` with lock-guarded entry.

### A4 — Blocking I/O inside the async WS loop · **Medium** · Sprint 2
**Description.** The WS handler writes to disk synchronously per message inside the asyncio read loop.
**Why it matters.** At 30–50 symbols × (kline + tickers) bursts, disk writes serialize the read loop → backpressure, latency, dropped frames.
**Recommended solution.** Buffer to an `asyncio.Queue`, batch-flush on a writer task (or thread executor); measure throughput under load.

### A5 — Reproducibility gap: configs referenced by path, not hashed · **Medium** · Sprint 2
**Description.** Manifests record `config_ref` as a path string; config/threshold *content* isn't hashed in.
**Why it matters.** A run can't be tied to the exact config bytes; editing `thresholds.yaml` bumps no version.
**Recommended solution.** Hash config/threshold/feature-spec contents (sha256) into every manifest; include the git commit.

### A6 — Observability is pull-only · **Medium** · Sprint 2
**Description.** Health/report are scripts you must run; nothing actively alerts when `overall_ok` flips or the collector dies; no metrics/tracing.
**Why it matters.** In production a silent collector death goes unnoticed until the dataset is stale.
**Recommended solution.** Push health transitions to the existing alert channel; export metrics (Prometheus/textfile); alert on data-staleness and calibration/PSI breaches.

### A7 — Layering smell: dataset layer imports model layer · **Low** · Backlog
**Description.** `datasets/build.py` now imports `MODEL_FEATURES` from `models/`.
**Why it matters.** Minor inverted dependency; acceptable now, a coupling risk as the codebase grows.
**Recommended solution.** Move the canonical feature list to a neutral module (e.g. `amber/common` or `features/spec`) both layers depend on.

---

## Board consensus & disagreements

**Consensus.** The build quality is good; the risk is concentrated in the **statistical** and **market-realism** layers. Two Criticals gate everything else: *is there edge on real data* (Q2) and *does it survive real fills* (T1). Until both are answered, further scaling and feature expansion are speculative.

**Recorded disagreement — Quant vs Architect on sequencing.** The Architect argued storage/scaling (A1) should be tackled early to avoid rework. The Quant objected that scaling an unproven-edge system is textbook premature optimization. **Resolution:** prove edge on real data first (Sprint 1); do only the *cheap correctness* infra now (A2 atomic writes, A3 lock); defer the storage rework (A1) to Sprint 3 when data volume justifies it.

**Recorded disagreement — Trader vs ML on order-flow (T3).** The Trader rates order-flow features essential and near-Critical; the ML lead cautioned against adding a heavy new data source before the current feature set is proven or falsified. **Resolution:** run Sprint-1 on the current features to get a clean read; if edge is marginal-but-present, T3 becomes the top Sprint-3 lever; if edge is absent, T3 is the first thing to try before abandoning the thesis.

---

## Priority matrix

| Sprint 1 — prove-or-kill + cheap correctness | Sprint 2 — validity & robustness | Sprint 3 — realism & scale | Backlog |
|---|---|---|---|
| Q2 real out-of-sample edge | Q3 overlap embargo/weights | T1 depth-based fill model | T6 cross-exchange |
| Q1 universe/survivorship logging | Q4 per-regime eval | T3 order-flow/liq features | A7 feature-list relocation |
| Q7 PR-AUC + reliability | Q5 threshold/feature decoupling | T4 wash-volume filter | |
| A2 atomic state writes | Q6 warm-up gating | T5 contract-state | |
| A3 single-instance lock | M1 coherent probabilities | A1 Parquet partitioning | |
| | M2 class imbalance | M6 artifact schemas | |
| | M3/M4/M5 feature hygiene & recalibration | | |
| | T2 entry-lag in backtest | | |
| | A4 async write buffering | | |
| | A5 config hashing · A6 push observability | | |

## The three that gate go-live

1. **Q2 — Prove edge on real data.** Everything is theoretical until a purged out-of-sample PR-AUC on real mainnet data clears a pre-declared bar.
2. **T1 — Make the backtest tradeable.** Depth-aware fills + capacity, or the P&L is fiction on exactly the coins that pump.
3. **Q1 — Kill survivorship/selection bias.** Sample the real event population, not a hand-picked survivor set, or the base rates lie.
