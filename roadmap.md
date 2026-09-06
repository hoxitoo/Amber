# Project Amber — Roadmap

_Last updated: 2026-08-26_

Amber is a local-first ML scanner for Bybit futures that predicts event
probabilities (pump/dump) and emits alerts. Not an auto-trader.

Legend: `[x]` done · `[~]` partial · `[ ]` planned.

---

## Stage 1 — Stable local ML scanner foundation · **done**
- [x] Local pipeline skeleton and runnable scripts.
- [x] Real Bybit v5 WS client (kline + tickers → bid/ask/OI/funding), REST backfill.
- [x] `NormalizedRow` strict validation (Pydantic v2), gap-fill + `is_synthetic`.
- [x] Idempotent ingestion (byte offsets + per-symbol ts watermarks; safe re-runs).
- [x] Unified `FeatureEngine` (offline/live parity, no train/serve skew).
- [x] Signal schema (`SignalV1`), manifests, artifact registry.

## Stage 2 — Modeling correctness and trust · **done**
- [x] LightGBM pump/dump dual-model (logreg fallback, constant-head for single-class).
- [x] Adaptive volatility-based labeling, multi-horizon, censored-row exclusion.
- [x] Time-based walk-forward CV with purge gaps; dedicated calib/test segments.
- [x] Isotonic calibration on a held-out segment; out-of-sample eval (`in_sample` flag).
- [x] SHAP-style feature contributions in explanations.
- [x] Breakout / momentum-precursor feature pack (18 features: volume surge,
      volatility squeeze, breakout geometry, OI rate-of-change).
- [x] Directional score + spread/cooldown/concurrency risk filters (persistent state).

## Stage 3 — Monitoring and validation · **done**
- [x] Health check on the paths the pipeline actually writes.
- [x] Rolling AUC over **confirmed real outcomes** (no self-referential labels).
- [x] Per-feature PSI vs train reference; prediction-bias monitor.
- [x] Model-driven event backtest on the test split (promotion gate).
- [x] PR-AUC + lift-over-base-rate + reliability curve in eval (audit Q7).

## Stage 4 — Production readiness · **done**
- [x] Telegram + Discord transports (env credentials), router warns on unknown channels.
- [x] Universe selection with liquidity floor + warm-up gating.
- [x] systemd units + docker-compose (`deploy/`); mainnet/testnet switch.
- [x] Atomic state writes + single-instance locks on all stages (audit A2/A3).
- [x] Warm-up-row gating in dataset and scanner (audit Q6).

## Stage 5 — Productization · **in progress**
- [x] Streamlit dashboard (status, live signals, model quality, data/drift, per-symbol).
- [x] Control panel: start/stop services, run pipeline stages, symbol editor — all in UI.
- [x] One-click launchers (`Amber.bat`, `launch.command`) + PyInstaller packaging.
- [x] CI: lint + 3.11/3.12 test matrix; Windows build workflow (downloadable package).
- [ ] Model registry rollback workflow from the UI.
- [ ] Parquet storage backend with date partitioning (audit A1).
- [ ] Push observability: alert on `overall_ok` flip / collector death (audit A6).

---

## Current state (honest)

Built and tested (285 tests), running 24/7 on real Bybit mainnet data over 27
symbols. Sprint 1's open question — *does the model have real predictive edge?*
— has a first real-data answer, and it is qualified rather than clean:

> **Measured 2026-09-06.** At a fixed 1% barrier over 15 bars with one-sided
> labels, precision was 58% against a 5.8% base rate: **lift 4.51** after a
> family-wise correction across 24 label definitions and after collapsing
> correlated alerts into 18 independent market episodes. The ordering of all
> eight ruler/shape combinations replicated identically across two horizons
> (Spearman 0.976), which is stronger evidence than any single interval.

Three qualifications, all tracked in Sprint 5 and none of them cosmetic:

1. **It is not established that the edge is directional** (D1). `range_atr_14`
   holds 32.4% of permutation importance, so the model is substantially a
   volatility forecaster, and a pure magnitude forecaster scores lift ≈ 2 on
   this label by construction. The share of 4.51 that is direction is unknown.
2. **It was measured in one market session** (D3) — 18 episodes inside ~7 hours.
3. **It was measured at 388 alerts/day** (D4), which is a firehose. Precision at
   a rate a person can actually act on has not been measured.

The previous label — a volatility-scaled barrier — ranked **last of 24** at lift
0.21–0.39, i.e. its alerts were indistinguishable from random. It had been live
for roughly ten days. See `docs/target_review_2026-09.md`.

See `docs/audit_review_board_2026-07.md` for the full institutional audit.

---

## Future steps — audit-driven backlog

### Sprint 1 — prove-or-kill + cheap correctness
- [x] A2 atomic state writes · A3 single-instance locks · Q6 warm-up gating ·
      Q7 PR-AUC + reliability · Q1 universe logging + `list_instruments`.
- [ ] **Q2 — prove edge on real data (owner action).** Collect ≥2–4 weeks of real
      mainnet data over a rule-based universe; read purged out-of-sample PR-AUC;
      declare a kill criterion up front. *Not a code task — needs real collection.*

### Sprint 2 — statistical validity & robustness · **code done**
- [x] Q3 uniqueness sample weights (1/horizon) for overlapping event windows.
- [x] Q4 per-regime evaluation (trend/range × high/low vol) in verbose eval.
- [x] Q5 label threshold lagged one bar (decoupled from the current bar's ret_1).
- [x] M1 coherent probabilities: joint normalization, p_none exposed in signals.
- [x] M2 class imbalance: scale_pos_weight (LightGBM) / balanced logreg fallback.
- [x] M4 winsorization: train-time clip bounds stored in the model, applied at inference.
- [x] T2 1-bar execution lag in the backtest (decision bar i → entry/outcome bar i+1).
- [x] A4 buffered WS writer (queue + batch flush off the read loop).
- [x] A5 sha256 of all configs embedded in every artifact manifest.
- [x] A6 push alert on overall_ok flip (Telegram text via report run).
- [x] M3 permutation importance + correlation pruning — measured out-of-sample
      each retrain (`logs/feature_importance.json`, shown on the Модель tab), so
      "which of the 21 features actually work" is answered with numbers rather
      than guessed at before spending on new inputs.
- [x] M5 rolling recalibration — calibration is checked against the freshest
      confirmed outcomes every `pipeline.recal_min` (default 20 min) and refit
      when ECE/bias drift past threshold. Calibration maps scores to an event
      frequency, and that frequency moved 10%->22% (pump) and 7%->28% (dump) in
      two weeks, so it goes stale long before the model does; refitting is two
      coefficients and needs no retrain.

### Sprint 3 — market realism & scale
- [~] T3 order-flow: taker aggressor imbalance / CVD from WS `publicTrade`
      (`taker_imbalance`, `cvd_norm_20`, `trade_count_z_20`) — **done**;
      order-book depth/imbalance (`orderbook`) and liquidations still pending.
- [ ] T1 depth-aware fills + capacity in the backtest (collect L2).
- [ ] T4 wash/manipulated-volume filter · T5 contract-state (ST/delisting) awareness.
- [ ] A1 Parquet partitioning · M6 artifact schema validation.

### Sprint 4 — signal transparency & outcome tracking · **planned (not started)**

Requested 2026-07-26. Make each signal self-explanatory and self-scoring: show
how confident the model is, and later whether it was right.

- [ ] **S4.1 — Per-signal confidence in the UI.** Surface the model's confidence
      for every signal in the "Модель" tab signal list, e.g. `pump — уверенность
      70%`. The number is the head's **calibrated** probability
      (`prob_up_calibrated` / `prob_down_calibrated`), which the model already
      produces — this item is primarily (a) displaying it clearly next to each
      signal and (b) making the calibration trustworthy enough that "70%" really
      means ~70% hit rate (reliability-curve check; recalibrate if the curve
      drifts). Show the direction's own probability, and optionally `p_none`.
      *Model-side note:* confidence is not a new model output — a well-calibrated
      probability IS the confidence. The work is calibration quality + honest
      display, not inventing a separate confidence score.
- [ ] **S4.2 — Realized outcome per signal.** After a signal's horizon elapses,
      resolve whether the predicted event actually happened (pump/dump hit vs
      miss vs timeout) and show the verdict next to the signal (e.g. `dump →
      сработал` / `не сработал` / `таймаут`). Persist the resolution so the
      signal log becomes a track record.
      *Infra that already exists to build on:* the quality monitor already
      lag-joins `signals.jsonl` with realized `normalized` labels after the
      horizon to compute rolling AUC over **confirmed** outcomes — S4.2 reuses
      that join to stamp a per-signal hit/miss and expose it in the UI, plus a
      running hit-rate summary (and, once volume allows, hit-rate broken down by
      confidence bucket to validate S4.1's calibration end-to-end).

### Sprint 5 — does the edge point anywhere? · **planned (not started)**

Raised 2026-09-06, after the label sweep replaced the target (see
`docs/target_review_2026-09.md`). The stated goal of the project is to *catch
volatility at its earliest stage and determine direction*. These items are the
gap between that sentence and what the system currently measures. Every one is
a known inaccuracy or dead end, not a feature wish.

- [x] **D1 — Separate "will it move" from "which way". `TOOL BUILT, AWAITING A LIVE RUN`**
      `scripts/run_label_decomposition.py` trains one head per factor and reports
      each separately. Validated against a fixture where a volatility burst is
      predictable and the sign is a fair coin: it reports `move` at lift 3.62
      (bound 2.42) and `direction` at 1.12 (bound 0.74), verdict
      `magnitude_only` — while the production `pump` target on that same data
      reads lift 3.64 with a *significant* bound of 1.77. That is the trap in
      one line: pump looks like directional edge and is entirely magnitude. The
      complement fixture, with the sign leaked into a feature, returns `both`.
      *Remaining:* run it on the live box and act on the verdict.
      The goal names two different questions and the label merges them.
      `up_hit` means "rose 1%", so a model that forecasts *magnitude* perfectly
      and knows *nothing* about sign still scores lift ≈ 2, because half of all
      large moves are up. M3 says this is not hypothetical: `range_atr_14`
      carries 32.4% of permutation importance, i.e. the model is largely a
      volatility forecaster. **An unknown share of the sweep's lift 4.51 is
      magnitude, not direction.**
      *Do:* label `move_hit` (|move| ≥ target, either sign) and `direction`
      (sign, conditional on a move having happened), run the existing sweep
      against each, and report two lifts instead of one.
      *Why first:* costs no new data, reuses `scripts/run_label_sweep.py`, and
      its answer decides whether D2/D7 are worth their cost. A likely outcome is
      that magnitude is strong and direction is near a coin flip — which is not
      a failure but a redefinition: "scanner of starting volatility" is honest,
      useful and reachable, while "direction predictor" would not be.

- [ ] **D10 — Come back for direction. `FUTURE — deliberately deferred`**
      Measured on live data 2026-09-06: direction is **not** predicted at all.
      `direction` scored precision 0.590 against a 0.587 base rate — the model
      adds **+0.3 percentage points** over naively saying "up" every time, and
      that 0.587 is itself just the market rising during the test window. The
      whole of the production target's lift 11.82 is explained by magnitude
      times that drift: 0.850 × 0.587 = 0.499 against an observed 0.530.
      **The product was therefore redefined as a volatility scanner**, and the
      direction call belongs to the person reading the chart.
      *This is a deferral, not a closed question.* Two things are worth being
      precise about:
      - With 12 episodes the measurement can only exclude a *large* directional
        effect (precision ≥ 0.90). A moderate one — 65–70%, still tradeable —
        is not excluded by the interval, although the point estimate gives no
        hint of one either. This needs D3's longer window to settle.
      - The pump/dump heads are **kept** rather than deleted, precisely so this
        can be revisited without rebuilding the machinery: they keep training,
        `scripts/run_label_decomposition.py` re-answers the question on demand,
        and only the *primary* signal moved to `move`.
      *Revisit when:* D3 gives 50+ episodes, or a genuinely leading input lands
      (D2/D7 — which are not worth buying for direction on today's evidence, but
      would change it if acquired for other reasons).

- [ ] **D2 — "Early stage" is unreachable on 1m bars.**
      By the time a 1-minute candle closes and `range_atr_14` registers a spike,
      the move is a minute old — mid-move, not early, on a venue where bots act
      in milliseconds. The three order-flow features exist but are aggregated to
      the minute and measured at 0.0016 / 0.0009 / 0.0009 importance, under 1%
      combined: they were flattened into uselessness by the bar.
      *Do:* consume `publicTrade` at tick level and L2 `orderbook` deltas —
      aggressor imbalance, depth imbalance, liquidation bursts, OI jumps — on a
      sub-minute clock. Overlaps T3's remainder and T1.
      *Gate:* only worth the collection and storage cost if D1 finds directional
      signal to sharpen.

- [ ] **D3 — Every result so far comes from one market session.**
      The adopted label was selected on 18 independent episodes inside a ~7-hour
      test segment. Crypto has strong intraday seasonality (Asia / Europe / US)
      and regimes lasting days. Lift 4.51 is a one-regime measurement.
      *Do:* re-run the sweep on a longer window (`--max-candles 6000+`) once
      history allows, and evaluate per session and per regime. Settles the
      horizon question too, which the 7-hour segment structurally cannot: at
      h=60 it admits at most 7 independent episodes.

- [ ] **D4 — The measured operating point is not an operating point.**
      Precision was compared at a 1% alert budget: **388 alerts/day across 27
      symbols**. That is a firehose, not something a person acts on. A usable
      rate is 10–20/day, and precision there has never been measured — it is a
      different, much more selective point on the same curve.
      *Do:* report the precision/recall curve down to realistic budgets and set
      the live thresholds from that, not from the measurement budget.

- [ ] **D5 — No time-of-day or market-regime context.**
      A pump at 03:00 UTC on a thin book and one in the US session are different
      events with different exit liquidity. Nothing in the feature set separates
      them.

- [ ] **D6 — No "what already happened" state.**
      P(+1% | coin already up 8% this hour) differs sharply from P(+1% | quiet).
      `dist_to_low_20` gestures at this and scores 0.0008 importance.
      *Do:* explicit run-up / drawdown-from-recent-extreme features over several
      lookbacks. Cheap: computable from data already collected.

- [ ] **D7 — Nothing checks whether a signal is tradeable.**
      The spread filter (30 bps) exists; book depth does not. An alert on a coin
      with $2 000 resting within 1% is not actionable at any size. Precedes T1
      and shares its L2 collection.

- [ ] **D8 — Prune the 9 dead features.**
      M3 measured 9 of 21 at or below shuffle noise, with the top five carrying
      97%. They contribute nothing and give the model surface to overfit.
      Cheap, and should follow D1 so pruning is judged against the right target.

- [ ] **D9 — Normalized history is re-parsed unbounded.**
      Costs roughly 12 s per pipeline cycle today and is the CPU ceiling that
      caps the universe near 100 symbols. Prerequisite for any expansion beyond
      the current 27.

### Backlog
- [ ] T6 cross-exchange lead/lag features · A7 feature-list relocation.
- [ ] Universe expansion to 100–200 symbols. Measured ceilings: memory ~50
      symbols at a 48h window, CPU ~100 symbols, the latter set by D9.
- [ ] API + multi-tenant observability (SaaS direction).
