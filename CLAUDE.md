# Rules for working on Amber

Read this first, every session. Each rule exists because breaking it already
cost this project a wrong conclusion, a silent failure, or a round trip with the
owner. The incident is named so the rule is not mistaken for general advice.

The owner runs this 24/7 and acts on what I report. A confident wrong answer is
worse than "I don't know yet".

Two halves: **rules** (sections 0–7) and **context that exists nowhere else in
the repo** (sections 8–11: the owner, the box, what has been measured live, and
the plan). Sessions start from a fresh container with no memory of earlier
chats; sections 8–11 are that memory. **Update section 10 at the end of every
session** — a result that lives only in chat is lost.

## 0. Before touching anything

- `git fetch origin && git status && git log --oneline -1 origin/main`.
  The VPS deploys from **`main`**. A reset container once left the working copy
  on an old branch **50 commits behind** with 7 features instead of 21; code
  written there would have been built against the wrong feature set.
- If deps are missing after a container reset:
  `pip install -r requirements.txt -r requirements-dashboard.txt ruff==0.15.22`.
- Baseline: `python -m unittest discover -s tests -q` and
  `python -m ruff check amber/ tests/ scripts/` must be green before I start.

## 1. When a core definition changes, find every consumer

The primary target became `move` on 2026-09-06. For weeks afterwards:
`evaluate_model` measured `pump`; permutation importance measured `pump`; the
backtest and the daily threshold sweep replayed the pump/dump gate; the
Telegram alert led with up/down probabilities; the dashboard's out-of-sample
panel showed pump; and rolling recalibration (every 20 min) checked only
pump/dump and, on a refit, **dropped the move head's calibration**, so the live
gate ran on raw scores until the next retrain. All fixed 2026-09-29. The sweep
could even one-click apply `prob_lift_min` 1.2-3.0 — a value that now opens the
move gate (live: 9.55). Each reported numbers about a head that decides
nothing, and I reported them as if they meant something.

- On any change to the target, label, feature set, gating head or threshold
  semantics: `grep -rn` the old name across `amber/ scripts/ tests/` and list
  every consumer — train, calibrate, eval, importance, recalibrate, backtest,
  tuning, scanner, quality_report, dashboard, analysis scripts.
- Each one is either updated or explicitly labelled as measuring the old thing.
  Nothing is left to silently describe the wrong object.
- Add a test that pins the consumer to the new definition.

## 2. No verdict on a point estimate

Incidents: the label sweep reported lift 1.78 on a pure random walk; with
per-arm 95% bounds it still said `edge_survives_lag` at 24 arms; the baseline
check said `model_adds_signal` on a difference of **one alert in 103**.

- Every "better / worse / edge / no edge" claim uses the episode-clustered lower
  bound (`_episodes`, `clustered_low`) with a family-wise correction
  (`family_z(n_compared)`). Alerts within one horizon of each other, on any
  symbol, are one observation.
- `clustered_low` is exact (Clopper-Pearson) on the measured rate. Until
  2026-10-04 every tool rounded `rate x episodes` to whole hits (rounding up
  made the clustered bound exceed the naive one) and used Wilson, which
  under-covers at a few hits on a rare base rate: 2 hits in 15 at 1.2% read as
  lift_lo 1.47, exact ~0.14. Both found when a null fixture "found" a
  precursor in a column of pure noise. Never round hits; never use Wilson for
  a verdict on a handful of hits.
- A winner must clear the loser's *achieved* value with its *lower bound*.
  Otherwise the answer is "indistinguishable", and I say so.
- State the episode count next to every precision. Under ~10 episodes, say the
  result is underpowered before saying anything else.
- A new statistical tool is validated on a null fixture (known answer: no edge)
  **at the real arm count** before it is run on live data. An 8-arm null test
  passed while the real 24-arm run failed.

## 3. Measure; don't extrapolate. Check two numbers are comparable

Incidents: I claimed a recompute would peak at ~1.7 GB on 27 symbols by
extrapolating two points — it plateaus at 174 MB. I claimed "the edge dies in
one bar" from eval precision vs backtest precision, which pool different heads
and populations; the proper measurement refuted it.

- Resource claims (memory, disk, CPU, time) come from a benchmark at production
  size (27 symbols, current row count), not from arithmetic on a small run.
- Before comparing two numbers, confirm same head, same label, same segment,
  same population. If not, don't compare them — or say they aren't comparable.
- Mark every statement as **measured** or **inferred**. Inferences are never
  stated as findings.
- A new data source reads as 0 for every bar before it was collected, which the
  model cannot tell from "nothing happened". Before widening a training window
  or comparing across a date, check the history of every feature covers it.
- When a measurement refutes something I wrote earlier — in chat, a docstring, a
  comment or a doc — correct it at the source. A false comment misleads later.

## 4. Before anything is deployed, check what can take the box down

The box is 3.9 GB RAM, 59 GB disk, four services (`amber-ws-collector`,
`amber-pipeline`, `amber-scanner`, `amber-dashboard`). It has already hit a
full disk, OOM-kills, and a 10-minute dashboard load.

For every change that ships, answer before pushing:
- **Memory:** peak RSS at production size, measured (see rule 3).
- **Disk:** bytes per row x rows per day; is anything unbounded?
- **CPU:** is a one-time cost triggered (spec bump → full feature recompute)?
- **Persisted state:** state files written by the *previous* version will be read
  by the new one. The liquidation deploy would have crashed with
  `KeyError: 'liq_short'` on old trade buckets.
- **External failure:** what if Bybit rejects or changes the input? A single
  subscribe request meant one bad topic could silently kill the candle feed.
- **Rollback:** does an older model/artifact still work with the new code?

## 5. Tests must be able to fail

- After writing a test for a fix, revert the fix and confirm the test fails.
  Several tests here passed first time; only reverting proved they guard
  anything.
- Tests reference live definitions (`MODEL_FEATURES`, `FEATURE_SPEC_VERSION`),
  never hand-copied lists or version strings — two such copies went stale.
- Synthetic fixtures can create the signal they are meant to test for. A drift
  that lands in `ret_1` (a model feature) made direction "predictable" in a
  fixture meant to have none. Check that a null fixture really is null.
- Never tune a threshold until a test passes. If a metric fails on a fixture
  where the answer is known, the metric is wrong, not the threshold.

## 6. Commands I give the owner

The owner runs them verbatim on the VPS as root. Every command block is fenced
as ```bash so the chat highlights it — the owner asked for this; unlabelled
fences render as plain grey text.
- Always from the project dir, as the service user, with the venv:
  `cd /opt/amber && sudo -u amber git pull`
  `cd /opt/amber && sudo -u amber /opt/amber/.venv/bin/python scripts/<x>.py`
  Plain `git pull` as root fails on ownership; plain `python` has no deps; a
  relative path from `/root` does not exist.
- Say which services need a restart and why.
- Say how long to wait before a result means anything (the training window is
  `max_candles_per_symbol` long; a new feature carries zeros until it rolls over).
- Give the exact lines to look for in the output, and what each outcome means.

- The dashboard's start buttons and systemd run the same services. Every
  service entry point takes a `SingleInstanceLock` (kernel flock), retraining
  locks `models/`, dataset builds lock `datasets/`; a new long-running entry
  point must do the same or a button can start a second copy.

## 7. Never

- Ask for, accept, or store API keys. Amber uses Bybit **public** streams only;
  the order book, trades and liquidations are all public. If the owner offers a
  key, decline and tell them to revoke it if already created.
- Present direction as a prediction. Measured 2026-09-06: 0.590 precision vs a
  0.587 base rate (roadmap D10).
- Declare a result I have not measured, or skip telling the owner about a
  mistake of mine that changed a conclusion.

## Project facts worth not re-deriving

- Primary target: `move` — |price| reaches a fixed 1.0% within 15 bars,
  one-sided labels. Pump/dump heads are kept only for the D10 revisit.
- Decision tools: `scripts/run_label_sweep.py`, `run_label_decomposition.py`,
  `run_operating_curve.py`, `run_baseline_check.py`, `run_ledger_report.py`.
  All read-only. The ledger report is the one that answers "does it pay".
- As of 2026-10 the move model is indistinguishable from one-feature rules
  (`range_atr_14`, `bb_width_20`) in the baseline check AND in the forward
  ledger (move hit 87.0% vs 87.4%); no fixed direction rule beats the fee
  (section 10). Everything available on 1m
  bars says "a move is already under way"; see roadmap D2/D11.
- `docs/target_review_2026-09.md` holds the reasoning behind the current target.

## 8. The owner

- Writes in Russian; answer in Russian. Code, comments, commits and docs stay
  in English, as the repo already is.
- Does not read the code. Runs my commands verbatim as root on the VPS and
  pastes the output back. Every round trip costs them hours to days, because
  most results need a training window to fill first. So: one message with
  everything needed, never a drip of follow-up questions.
- Goal, in their words: catch volatility at its earliest stage and determine
  direction; precision is the main criterion. What they actually want to know
  is **whether acting on Amber's alerts makes money** ("выходить на прод,
  чтобы тестировать на успешность модель", 2026-09).
- 2026-09-29, verbatim concern: we keep finding and fixing errors instead of
  getting closer to testing the model for profit. That is fair. Every session
  must advance section 11 or say plainly why it could not. A fix that changes
  no number section 11 depends on goes to the backlog, not into the session.
- Once offered a Bybit API key; declined (rule 7).
- The VPS deploys from `main`, and all work is pushed to `main`. A harness may
  assign a `claude/...` branch; the owner never pulls those.

## 9. The box

- VPS, repo at `/opt/amber`, owned by user `amber`, venv `/opt/amber/.venv`,
  data under `/opt/amber/data/` (paths in `config/amber.yaml` are relative).
- 3.9 GB RAM, 59 GB disk. systemd services: `amber-ws-collector`,
  `amber-pipeline` (normalise → features → hourly dataset + retrain, see
  `pipeline.retrain_min`), `amber-scanner`, `amber-dashboard`.
- Logs: `journalctl -u amber-pipeline -n 200 --no-pager`. Memory during a
  retrain: `free -m`, `systemctl status amber-pipeline | grep -i memory`.
- Measured on the box 2026-09-26, after the liquidation deploy: 585 MB used
  in total at rest, disk 27%, spec-v5 feature recompute peak 174 MB.
- The retrain is the largest memory consumer. At the 72h window it peaked at
  2.24 GB **in a synthetic benchmark here**, not yet observed on the box.

## 10. Live results log — measured on the box, newest last

Update this at the end of every session. Date, what ran, the number, the
episode count, and what it does and does not show.

- **2026-09-06** — label sweep adopted `move`: fixed 1.0%, one-sided,
  h=15 (`docs/target_review_2026-09.md`). D1: direction precision 0.590 vs
  base 0.587, i.e. no directional skill → the product became a volatility
  scanner; direction is the human's call.
- **2026-09-08** — D4 operating curve → `prob_lift_min: 9.55`
  (`config/thresholds.yaml`, commit 7b6636a).
- **2026-09-26** — D11 first live run: model precision 1.000 vs
  `range_atr_14` 0.990, alert overlap 63%. I first reported
  `model_adds_signal` on a one-alert margin — wrong; the corrected verdict is
  "matched by range_atr_14". Same day: liquidation collection deployed
  (e138980, 61c15ec); verified 4 subscribe acks ok, all 27 symbols on spec v5.
- **2026-09-29** — D11 after 48h of liquidations: overlap with `range_atr_14`
  fell **63% → 28%**; precision model 0.922 vs `range_atr_14` 0.971;
  **8 episodes** (under MIN_EPISODES 10 → underpowered). Verdict
  `matched_by:range_atr_14`. The overlap drop says the model now ranks alerts
  by something other than ATR; whether that is liquidations is **not
  measured** — importance was computed on the pump head until 3ff3415.
- **2026-09-29** — pushed, **not yet deployed**: window 48h → 72h, split
  60/15/25 (efdf724), ~20 test episodes expected (inferred). Deploy only once
  liquidation history on the box is ≥72h (section 3: pre-collection rows read
  as "no liquidations"). Awaiting from the owner: `run_baseline_check.py`
  output and the move-head importance table from the first retrain after.
- **2026-09-29** — forward ledger built (section 11 step B), **not yet
  deployed**; ships in the same pull as the 72h window, so the ledger starts
  clean on the new config. Measured here at production size (27 symbols,
  100k-candle files): a normal cycle adds ~7 MB and 0.2 s; a 2-day-outage
  backlog of 600 alerts adds ~180 MB for ~2 s, then frees it. Disk: ~0.33 MB/day
  at the ~400 alerts/day the operating point was set for (~120 MB/year);
  ≤3 MB/day if the gate were saturated. `read_tail` now seeks from the end —
  it used to read each months-long candle file whole.

- **2026-09-29** — full-system audit, fixes pushed (not yet deployed; same
  pull as the window and the ledger): recalibration kept/checks the move
  head; backtest replays the move gate (momentum/fade, like the ledger);
  threshold sweep refuses move models; alert text leads with P(move) and says
  direction is not predicted; dashboard shows move metrics; ledger enters
  after the alert was actually sent (`emitted_ms`), not one bar after its bar;
  service/retrain/build locks, flock-based so a reboot cannot leave a service
  refusing to start. **Unknown until the owner checks:** whether the live gate
  has been running on uncalibrated move scores (latest calibration lacking a
  `move` head).

- **2026-10-03** — all of the above deployed on the box (owner's log,
  10:45-13:22 UTC). Measured: the ledger scores alerts every cycle, ~114 in
  155 min across both sources (~1 per 1.4 min); the shadow rule refits each
  retrain at `range_atr_14 >= 0.0053-0.0056`; the model fires on **3.4-4.5% of
  test rows**, about 3-4x the ~1% (~400/day) the D4 operating point was set
  for (inferred from 400 / (27 x 1440)) — left alone under the freeze, the
  ledger measures it as it is. Rolling recalibration now covers move: 12:35
  refit `move ECE 0.054 -> 0.009, bias +0.046 -> 0.000` (the model was
  over-promising moves by 4.6 pp on fresh rows). Because alerts are near
  continuous, 30 episodes would arrive in under a day from one regime, so the
  verdict now also needs **7 distinct UTC days** (`ledger.MIN_DAYS`), set
  before any ledger outcome was read. Not yet received: ledger report,
  retrain memory peak on the box, `run_baseline_check.py`, move importance.

- **2026-10 (4 days after the 2026-10-03 deploy)** — first full readings.
  *Ledger* (4 days, so formally underpowered until 7): model 999 alerts / 131
  episodes, rule 1188 / 135. Move hit rate **87.0% vs 87.4%** — model and
  `range_atr_14` rule identical. Net per trade, bps (mean / family-wise
  lower bound): model momentum -10.8 / -22.8, model fade -9.1 / -14.4, rule
  momentum -8.6 / -17.5, rule fade -11.1 / -17.4; win rates 48-49%. Momentum
  + fade ≈ -2 x cost (-19.9 and -19.7 vs -18), i.e. gross directional result
  ≈ 0: a coin flip that pays the fee. Inferred, not yet the formal verdict:
  with SE ≈ 5 bps no rule can reach a lower bound above 0 by day 7.
  *Baseline check* (72h window, 0.71-day test segment): model precision
  0.975 (20 episodes) vs `range_atr_14` 0.996 (13) and `bb_width_20` 1.000
  (14); overlap with ATR 49%; verdict `matched_by:bb_width_20`.
  *Importance, move head*: `range_atr_14` 79.6%; liquidation features
  `liq_count_5` 0.47%, `liq_imbalance_15` -0.02%, `liq_share_5` -0.08% —
  **liquidations add nothing measurable** (D2a closed, negative).
  *Memory*: `systemctl status` shows 2.2 GB for amber-pipeline (cgroup
  current, includes page cache; systemd here prints no peak). No OOM.
  Conclusion: on 1m public bars Amber detects "a move is under way" exactly
  as well as a one-line volatility rule, and neither fixed direction rule
  makes money. Next step is the owner's choice (section 11, D).

- **2026-10-04** — the owner rejected "Amber as a volatility indicator": the
  purpose is "price is calm now; P = x% of a sharp move soon, because of
  factors ...", i.e. warning BEFORE the move. The breakout-rule ledger idea
  was not built: it measures trading after a move starts, which is not that
  purpose. Built instead: `scripts/run_ignition_check.py` (read-only), which
  keeps only calm bars and asks whether anything predicts a 1% move from
  there (section 11, E). Validated: 10/10 null markets and 3 production-size
  nulls -> `no_precursor`; 5/5 planted -> found and named. Measured here at
  production size: peak 651 MB, ~12 s; it holds the retrain lock so the two
  never run together. The same validation exposed the bound bugs in rule 2;
  fixed in all four tools. Effect on earlier live results: bounds can only
  get stricter, so "matched_by" baseline verdicts stand; the 09-06 label
  sweep ranking and the 09-08 operating point (9.55) were chosen with the
  old bounds and have not been re-run (frozen; re-run before relying on them).

- **2026-10-04, first live ignition run** (72h window): calm share 23-30% of
  bars; base rate of a 1% move from calm **0.000-0.005** (≈1 in 1000 calm
  minutes at 30/15). All four arms printed `no_precursor` — **wrong reading,
  my tool's fault**: the 18 h test segment held only ~10 such moves, too few
  to exclude anything but a very strong precursor; the gate counted alert
  episodes, not events to predict. Point lifts (11.8, 4.9, 5.9) on 1-2 hits
  are noise; the printed factor table (sums > 100%) was noise too. Fixed:
  `underpowered` unless ≥20 independent moves in the test segment, factors
  shown only with a proven signal, and `--days` (default 30) reads history
  far beyond the training window from compact float32 arrays. Measured here
  at 30 days x 27 symbols (1.3 GB of feature JSON): peak 491 MB, ~9 min, so
  one hourly retrain is skipped while it holds the lock. Re-run pending.

- **2026-10-04, ignition run on 30 days** (182 h test segment): moves from
  calm per arm 67 / 114 / 22 / 41 independent (30/15, 30/30, 60/15, 60/30).
  30/30: model precision 0.162, lift 9.60, lift_lo 4.59 (162 alert episodes);
  best single rule `+range_atr_14` lift 8.69, lift_lo 3.99 ->
  `signal_matched_by:range_atr_14`. 30/15: rule only (lift_lo 2.30). 60/30:
  matched by `-dist_to_high_20`. 60/15: nothing. Model factors: range_atr_14
  65%, spread 14%, bb_width 8%; OI, funding, flow, liquidations near zero.
  Reading: a real, family-wise-significant effect, but the "precursor" is
  `range_atr_14` = mean candle high-low, while "calm" was judged on per-minute
  bid/ask mid snapshots. Flat snapshots over swinging candles is volatility
  already under way — the same signal the live model uses — not a warning
  before the move. Fixed the definition: every 20-bar slice of the window must
  also have a candle high-low range < 0.5% (recovered exactly from
  dist_to_high_20 / dist_to_low_20). Re-run pending; that run is the clean
  answer to the owner's question.

- **2026-10-06, ignition run with the candle filter** (30 days, 174-178 h
  test): moves from calm 44 / 87 / 17 / 40. 30/30: model lift 7.50, lift_lo
  2.56; `+range_atr_14` lift 7.50, lift_lo 2.22 -> `signal_matched_by`. 30/15:
  rule only (lift_lo 1.12). 60/30: none. 60/15: underpowered. Factors:
  range_atr_14 55%, spread_bps 17%, the rest ≤3%. Reading: the effect
  survives true calm, but its two carriers are mostly COIN properties (a
  jumpy alt's calm candle is wider and its spread wider than BTC's), so it may
  say which coin moves, not when. Added controls: `coin_only` (a static
  per-coin number) and within-coin ranking (same share of alerts from every
  coin) for every feature and for the model; verdicts `timing_signal` /
  `coin_choice_only`. A z-score version of the within-coin control leaked
  coin choice on noise fixtures (heavier tails win the alerts) — replaced by
  ranks before any live use; a property test pins equal per-coin shares.
  Measured here at 30 days x 27 symbols: peak 314 MB, ~13 min (with the
  test suite running beside it). Re-run pending; it decides between
  retargeting and D2.

- **2026-10-06, ignition run with coin controls** (30 days): verdict
  **`coin_choice_only`**. 30/30 (87 moves): model lift 7.50 / lo 2.35, rule
  `+range_atr_14` 7.50 / 2.05, static per-coin number 7.89 / **1.27**; within
  coin: model lift 4.08 / lo 0.72, best feature `rank:+range_atr_14` 5.39 /
  0.95. 30/15, 60/30: no_precursor; 60/15: underpowered. Reading: on 1m model
  features the effect is which coin moves, not when. The within-coin point
  lifts (4-5) are suggestive, but their bounds sit below 1 — not shown, not
  excluded.
- **2026-10-06** — the owner asked what could change the picture (other
  coins, longer candles, new data). Answer given: top-27-by-volume would make
  it harder (most efficient markets); longer candles hide a 15-30 min move,
  but an hour-scale question is worth asking; cheapest new information is
  already on disk (BTC/ETH lead, hour-scale build-up). Built into the
  ignition check before any of it was run: lead features (btc_absret_1/5,
  btc_range_atr_14, eth_absret_5), hour features (4 h range, 4 h / 24 h
  compression, 4 h return, last-hour notional vs 24 h), arms m30
  (all / liquid / thin by median spread), m60, and h6 (calm 6 h < 1.5% -> 3%
  within 4 h); the 15-bar horizons dropped (power: 17-44 moves). Fast
  rolling windows, pinned to the row-by-row labels by an equivalence test.
  Fixtures: 10 null and 8 jumpy-coin markets -> no timing signal; planted
  oi_roc_5 -> found 5/5; BTC jumping 2 bars before alt bursts -> found 3/3 as
  `timing_signal:rank:+btc_absret_1` (needs ~2 weeks of test data at z 3.8).
  Measured here at 30 days x 27 symbols incl. BTC/ETH: peak 406 MB, ~13.5
  min. Run pending. Next after this: sub-minute trade features, then the
  order book.

- **2026-10-09, ignition run with lead/hour features** (30 days, z 3.79,
  655 comparisons): first **`timing_signal:model`**. m30/all (94 moves, base
  1.24%): model within coin lift 5.65, **lo 1.39**; m30/liquid (51 moves):
  lift 7.70, lo 1.51. No single feature clears it within coin (best
  `rank:-ret_60` lo 0.14): the timing information is in the combination.
  Coin choice is still the larger part: globally model lift 7.64 / lo 2.46
  vs static per-coin 10.41 / 2.50. m30/thin, m60: no_precursor; h6:
  underpowered (13 moves). Factors (global PR-AUC drop, mixes coin and time):
  range_atr_14 52%, hr_range_240 25%, bb_width 7%, hr_notional_60_1440 6.5%,
  hr_compress 6%, hr_absret_240 5%, spread 4%, btc_range_atr_14 3.5%. The
  previous run without the hour/lead features had model-within lo 0.72, so
  the hour-scale features are what lifted it. Reading: "this coin was
  turbulent over the last hours, has been calm for 30 min" precedes a 1%
  move ~7x more often than its own base. Precision at the 1% budget is
  ~9.5% (inferred ~75 alerts/day, ~7 followed by a move). Caveats: one run;
  the ignition check has been re-specified several times (candle filter,
  coin controls, new features), and multiplicity ACROSS runs is not in z.
  Needs forward confirmation on data the check never saw before any live
  retargeting.

- **2026-10-09** — owner approved the live ignition channel. Built
  (`amber/signals/ignition_live.py`), **not yet deployed**: daily m30 model
  (30 days; earliest 85% fit, latest 15% Platt + per-coin top-1% threshold,
  i.e. "within coin" live), scored every scan on each coin's newest bar with
  the SAME `calm_now`/`feature_vector` as the check (parity test on 60 bars);
  every calm bar recorded with alert/notify (30-min pause per coin), resolved
  by the generalised ledger into `ignition_ledger.jsonl`; forward verdict
  `summarize_ignition`: alerts vs the hits expected from calm bars of the same
  coins, one comparison, one-sided 95%, needs ≥7 days and ≥20 independent
  moves (fixed before any record existed). Telegram off (`ignition.telegram:
  false`). Measured here: daily training at 30 days x 27 symbols peak 423 MB,
  ~9.5 min (normalisation pauses that long once a day; collector buffers);
  scan hook 1.3-2.4 s per minute, scanner +134 MB; record 263 B, ~7,800 calm
  bars/day (inferred) -> ~2 MB/day records + ~3 MB/day ledger, unbounded:
  revisit retention when the verdict is in. Dashboard reads a summary the
  pipeline publishes every 30 min, never the full ignition ledger.
  Review before deploy found one bug: the pipeline writes features symbol by
  symbol, so a scan could see an alt's new minute before BTC's and read the
  lead features as 0 where training always had values. Such coins now wait
  for the next scan; a test fails without it.

- **2026-10-09, ignition channel live on the box**: first model trained
  17:24 UTC (artifact 280 KB); scanner scores ~8 calm coins per minute, 0
  alerts at the first reading (expected: top 1% per coin); ignition ledger
  resolving from 17:57 (30 min after the first records), 1-4 bars per cycle.
  Open check: whether the ledger keeps pace with ~8 records/min (owner to
  compare `wc -l` of ignition_calm.jsonl vs ignition_ledger.jsonl an hour
  later; a growing gap > 600 lines means it falls behind). Earliest verdict
  2026-10-16. Logging made explicit (d188867) after an empty grep read as
  broken.

- **2026-10-13, first forward ignition reading** (~4 days live, 5 distinct
  UTC days because 10-09 counts): 25,412 resolved calm bars, 61 independent
  moves from calm; 200 threshold crossings (110 would have been shown, ~22
  per day), 53 alert episodes. Move within 30 min after an alert **8.00%**
  (~16 hits) vs **1.58%** expected from calm bars of the same coins: lift
  5.06, exact one-sided lower bound **1.83** (reproduced here from the
  printed numbers). Verdict `underpowered` only because of the pre-registered
  day floor (move floor already met). Consistent with the check's within-coin
  lift 5.65 (lo 1.39). Not comparable: the alert text's "обычно" (training
  base, bid/ask mid) and the report's 1.58% (forward base, candle wicks).
  Direction still not predicted. Ledger pace: 25k resolved in ~4 days is
  plausible but not checked against `ignition_calm.jsonl` (still asked).
  Because `days` counts distinct UTC days, the day floor is met on 10-15,
  not 10-16 as written above; the rule stands as fixed.

- **2026-10-17, ignition forward verdict `confirmed`** (9 distinct UTC days,
  ~7.3 days live): 40,302 resolved calm bars, 106 independent moves from
  calm; 574 threshold crossings (277 would have been shown, ~38/day,
  inferred from the span), 133 alert episodes. Move ≥1% within 30 min after
  an alert **5.92%** vs **1.38%** for calm bars of the same coins: lift 4.29,
  exact one-sided lower bound **2.14**. Pre-registered single comparison, on
  data the check never saw, so the across-run multiplicity caveat of 10-09
  does not apply to it. Precision fell from 8.0% at day 4 (normal: early
  reads run high), the bound rose with more data. Ledger pace measured:
  `ignition_calm.jsonl` 40,833 vs `ignition_ledger.jsonl` 40,626 lines, gap
  207 ≈ the last 30-min horizon still open — it keeps up. What it does NOT
  show: profit. ~94% of shown warnings are followed by no 1% move, and
  direction is not predicted; whether any trade on these warnings beats the
  fee is unmeasured. Precision is reported over all threshold crossings, not
  over the shown subset. One month, one regime. Telegram still off; the
  owner decides (section 11 F). The 2026-10-16 conversation also raised a
  second mode (4H/1D pullback-to-support on BTC/ETH); proposed as a
  read-only historical backtest first, not started.

- **2026-10-17, Telegram on + bracket money test built** (pushed, deploy
  pending). `ignition.telegram: true`; the owner created a bot, credentials go
  in `/etc/amber/alerts.env` (root, 600; never shown to me). Only ignition
  warnings go to Telegram (`alerts.channels` still console only), ~38/day
  (inferred). One failed send ends that scan's sending, so a Telegram outage
  cannot hold the scan for minutes. Money test, fixed before any outcome:
  breakout **bracket** on every shown warning — stop orders +-0.3% from the
  entry bar's close, the first fill is the trade and the other order's level
  its stop-loss, target +-1% from the close, 30-bar window, unfilled = no
  trade; costs **15 bps** round trip (both sides taker, Bybit base 0.055% a
  side, + 4 bps slippage), gross stored too. Verdict `summarize_bracket`:
  episode-clustered mean net per trade, one comparison, one-sided 95% lower
  bound > 0, ≥30 episodes and ≥7 days. Same-day review caught my first
  version (stop back at the close) overstating results: on 1m bars a dip
  before the breakout cannot be told from a return after it, worth 0.00 bps
  calm but +0.8 / +3.2 / +6.4 bps a trade at 0.3 / 0.5 / 0.8%/min, i.e.
  exactly when brackets fill. Moved the stop to the other order before any
  outcome was computed (a00a0af); now no simulated trade scores better on
  bars than on its sub-minute path (tested at 0.12% and 0.5%/min), and a
  random walk at the booked cost reads `not_shown_profitable`. Booking stops
  at their exact level instead of where the path crossed is ~2 bps (what the
  slippage allowance covers). Telegram: values in alerts.env are stripped of
  CR/quotes (Windows-made file), a 4xx is not retried, the token is redacted
  from transport-error logs. Warnings resolved before the
  bracket existed are scored once by `scripts/backfill_ignition_bracket.py`
  (same entry bar, checked; separate file; ledger never rewritten). Measured
  here at 27 symbols x 100k-candle files, 60k ledger rows: peak 224 MB, 6 s.
  Telegram text now gives the bracket levels from the bar's mid price (the
  ledger uses the next bar's close; a few bps apart).

- **2026-10-17, bracket verdict `not_shown_profitable`** (deployed on the box
  same day: pull fccb5a6, Telegram test send `True`, scanner + pipeline
  restarted; backfill over the 10-09..10-17 warnings, rule fixed before any
  outcome): 277 shown warnings scored, 198 filled (71%), win rate 24%, **102
  episodes over 8 days**. Net per trade **-16.1 bps**, lower bound **-22.9**;
  total -32% at one position per trade. Gross = net + 15 bps (constant cost)
  = **-1.1 bps**, inside about +-4 bps of zero (SE inferred from the bound):
  even at zero fees the bracket would not be shown profitable. Same picture
  as the main ledger's momentum/fade: a fill at +-0.3% is a coin flip on
  which way it goes next, and the fee decides. Reading: the warning is real
  (more 1% moves than calm bars of the same coins) but only ~6% of warnings
  are followed by one, and a symmetric stop rule cannot pick the side. Per
  section 11 G the bracket is not re-tuned on these data; the ledger keeps
  booking it forward. The Telegram text no longer shows bracket levels; it
  says the warning is not a trade signal.

- **2026-10-17, shown-probability check built; docs brought up to date.**
  The owner asked to show "уверенность в движении" (e.g. 30%) on warnings.
  Answer given: the message already shows the calibrated P; a "30%" would be
  invented (forward rate after warnings is ~6%), and a P is worth showing
  only if it is honest forward. `summarize_calibration` (in the ignition
  report): mean shown P of the warnings a person sees vs the
  episode-clustered exact two-sided 95% interval of what followed (wick
  touches, which can only run at or above the mid-based label), verdicts
  `consistent` / `overstated` / `understated`, ≥30 episodes and ≥7 days;
  plus a descriptive table by P band over all calm bars. Fixtures: honest P
  -> consistent 9+/10 seeds; 12% shown vs 4% real -> overstated; 2% vs 10%
  -> understated. Not yet run on the box. README / context.md / roadmap
  updated to the October state (they still described July-September);
  codex.md and plan.md marked historical. Also reviewed for the owner: the
  Telegram channel t.me/ftcrypta (level breakouts after 2-7 touches on
  15m/1H/4H, no stop/target, results shown to the extreme in hindsight, Bybit
  referral in signal posts; on two charts a "Шорт" position marker sits
  exactly at an advertised "long above" level — inferred, not proven).
  Owner approved checking that channel's signals on candles, **last** in
  the queue (section 11 I).

- **2026-10-17, shown-probability verdict `consistent`** (box, 9 days):
  shown warnings 277, promised on average **5.3%**, a 1% move followed
  **7.2%** (two-sided 95%: 3.4-13.2%, 127 episodes). By P band over all
  40,945 calm bars (mean P -> what happened): 0.4 -> 0.5%, 1.4 -> 1.8%,
  2.9 -> 2.8%, 5.1 -> 6.8%, 8.8 -> 11.5% (87 bars, 38 episodes): it ranks,
  and runs slightly above P at the top, as wick touches should. Ignition
  lift now 4.37, lower bound 2.19. **Open, flagged to the owner:** the report
  listed one shown warning (01:13 UTC) in its last 5,000 records, and the
  scanner journal showed `0 alerts, 0 shown` on every scan 09:42-12:49 UTC,
  against ~38 shown a day before — about 14 h with one warning. Saturday may
  explain part; a daily model whose thresholds no longer cross would explain
  all of it. Diagnostic (alerts per day by model_run_id, artifact times)
  asked. Also measured from journal timestamps: scans run every ~103 s, not
  60 (`time.sleep(interval)` after a ~43 s scan), so ~40% of minutes are
  never scored — no bias in the verdicts (warnings and their base come from
  the same scanned bars), fewer warnings; backlog.
  **Resolved the same day — not a fault, a weekend effect** (owner's
  diagnostic, alerts as a share of calm bars per UTC day): Fri 10-09 0.21%
  (partial day), Sat 0.04%, Sun 0.15%, Mon 1.91%, Tue 4.39%, Wed 5.63%, Thu
  5.15%, Fri 0.99%, Sat 10-17 0.02%; overall 1.39% against the 1% budget.
  Weekends have 2-3x the calm bars and almost no hour-scale turbulence, which
  the model keys on, so Telegram goes nearly silent on Saturdays and Sundays.
  The 10-16 model (`b0dc3c95`) does alert (01:13 on 10-17). Daily models
  retrain ~18:00 UTC. Watch: `coins_with_threshold` fell 24 -> 23 -> 19 of
  27 over 10-14..10-16 (a coin needs >= 100 calm bars in the latest 15% of
  the window); coins without one never alert.

- **2026-10-17, pullback study (section 11 H) — no variant pays.** Built
  `amber/backtest/pullback.py`, `history.py`, `scripts/run_pullback_check.py`
  (rules in the module docstring, fixed before real data). Validated first
  at the real arm count and length (12 arms, 2 correlated symbols, 6.7 years
  of 4H): zero-cost null -> no profitable arm in 6/6 seeds; pure trend ->
  long arms profitable but `matched_by_trend` in 6/6; planted bounce ->
  found (`beats_trend`). Per-trade look-ahead tests (flat future after entry;
  a limit fill blind to its own close and far extreme); mutations caught (a
  pivot "peek" was not, correctly: a pivot cannot be broken before it is
  confirmed). Run here on the public Binance USD-M archive (Bybit's API is
  geo-blocked from this container), BTC + ETH 2020-01..2026-09, funding
  included: **all 12 arms `not_profitable`** at z 2.64. Best: drop/limit/long
  +29 bps net per trade, lower bound -38, 217 episodes, control (random
  entry in the same trend) +22; retest/confirm/long +12 vs control +18; win
  rates 26-36% at 3:1. Point estimates of the long arms sit at their
  control: whatever they earn is the 2020-26 bull trend (buy and hold +804%
  BTC, +1313% ETH), not the support. Halves disagree in sign for most arms.
  SE ~25 bps a trade, so only an edge above roughly +65 bps is excluded.
  Measured here: peak 206 MB, ~8 s compute after a ~3.5 min first download
  (4.5 MB cache in data/history/). Bybit cross-check on the box optional.

- **2026-10-17, end of session.** Pump-fade idea parked in the backlog (11 J,
  roadmap D12). The owner asked whether simply waiting ~3 weeks makes the
  model understand the market better. Answer given: no, by design — the
  ignition model retrains daily on the last 30 days (`ignition.train_days`)
  and the move model hourly on the last 72 h, so neither accumulates
  knowledge; it adapts to the recent month. Waiting does narrow every
  forward bound (~3x the episodes -> ~1.7x tighter, inferred) and shows
  whether the ignition edge holds across regimes, but it will not turn the
  bracket (gross ≈ -1 bps vs 15 bps cost) into profit. The one thing that
  needs calendar time is D2 data (order book / sub-minute flow) — collection
  would have to start before the wait to be testable after it. Offered; not
  started. Widening ignition training to 60+ days is possible once feature
  history covers it, but it changes the confirmed model (new forward test,
  memory to measure first).

- **2026-10-17, D2 collection built** (pushed, deploy pending): the owner
  chose to collect order book + sub-minute flow during a ~3-week wait.
  `amber/exchange/orderbook.py`, `amber/pipeline/book_app.py`,
  `scripts/run_book_collector.py` (service `amber-book-collector`, own
  socket/process/lock `book_collector`, dashboard button), `scripts/
  run_book_report.py`. Subscribes `orderbook.200` + `publicTrade` for the 27
  symbols; every 5 s one CSV row per symbol: mid, spread, L1 size, resting
  notional within 5/10/25/50/100 bps each side, depth reach, buy/sell flow
  (bucketed by trade time), book age. Rows go to `data/raw/book/SYM/day.csv`;
  finished days gzipped (appending, never overwriting), >45 days deleted,
  nothing written under 2 GB free; if no message arrives for 10 s the book
  fields are left empty (socket down). systemd caps: CPUQuota 50%,
  MemoryMax 400M, Nice 10, Restart on-failure (`book.enabled: false` exits
  0). Measured here against a local fake Bybit (real Bybit is geo-blocked
  from this container, so real message rates are NOT measured): ~405 msg/s
  (inferred real load: orderbook.200 every 100 ms + trades) -> 2.9% of one
  core, RSS 27 MB; 10x load (5,400 msg/s) -> 29%. Rows ~185 B -> ~85 MB/day
  raw, gzip x3.5 on synthetic rows -> ~25 MB/day; 45 days ≈ 1.2 GB (inferred
  upper). The service logs its real msg/s, CPU, RSS and MB/day every 10 min.
  Review before push found and fixed: rows from a dropped socket kept
  writing the stale book; a clock step back could duplicate an interval;
  trades of a skipped interval vanished silently; a day re-gzipped could
  overwrite its archive. Each has a test that fails without the fix; one
  benchmark of mine was wrong first (27x the intended rate, read 65% CPU) —
  caught by profiling before any number reached the owner.
  **Deployed on the box 2026-10-17 14:28 UTC**: 27 symbols, 54 topics,
  `WS subscribe ok` for orderbook and publicTrade. **First stats, measured
  on the box** (Saturday 14:28-14:38 UTC): 183 book + 32 trade msg/s (about
  half the 405 I inferred), CPU **7% of one core** (the box runs ~4-5x the
  CPU per message of my local benchmark, 2.9% at 405 msg/s), RSS 26 MB
  (cgroup 16 MB), 3,240 rows = 27 x 120 intervals exactly, **~63 MB/day
  raw** (-> ~18 MB/day gzipped, ~0.8 GB at 45 days, inferred from the
  synthetic x3.5), stale 0, crossed 0, late trades 0. Weekday message rates
  are likely higher (inferred); the 50% CPU cap leaves ~7x headroom over
  today. Awaiting a 1-hour `run_book_report.py`. Direction check due
  ~2026-11-07.

## 11. The plan to a profit test

Proposed 2026-09-29. Why we have been circling: the only out-of-sample data is
the test segment of a rolling window — 7h, soon 18h — so every result is
underpowered, every run ends in "wait for more data", and the wait gets filled
with fixes. A rolling window can never accumulate evidence. What can:

**A forward ledger.** Every live alert, scored after its horizon and kept
forever. It is out-of-sample by construction, grows every day, and is the only
thing that can say whether acting on alerts makes money.

- **A. More test episodes** — done (efdf724), awaiting deploy.
- **B. Build the forward ledger** — built 2026-09-29: `amber/monitoring/ledger.py`,
  `amber/signals/shadow.py`, `scripts/run_ledger_report.py`, dashboard panel
  "Журнал сделок". Started on first run at the end of the existing logs.
  Barriers are checked on candle highs/lows (where a resting order fills);
  the training label follows the bid/ask mid, so the ledger's move hit rate
  is not comparable with calibrated P(move) — compare model with rule inside
  the ledger. Per alert: ts,
  symbol, calibrated prob, `model_run_id`, did |move| ≥1% happen within 15
  bars, first-touch direction, and the net result (fees 0.09% round trip, TP =
  SL = 1%, timeout 15 bars, entry on the close of the bar after the alert
  bar, a bar touching both barriers booked as a loss) of rules **fixed before any data is seen**:
  *momentum* (trade the direction of the alert bar) and *fade* (the opposite).
  Same ledger for the `range_atr_14` rule at the same alert rate, so the model
  is always judged against the indicator. Shown on the dashboard.
- **C. Freeze** code, features, labels and thresholds while the ledger fills.
  Hourly retraining continues (it is part of the system); `model_run_id`
  records which weights fired. Only bugs that change ledger numbers get fixed.
- **D. Decide** at ≥30 independent episodes spanning ≥7 distinct UTC days
  (rule 2 bounds, family-wise over the rules compared; the day floor because
  live alerts are near continuous and 30 episodes can come from one day): if a rule's net result has a lower bound above 0, the
  owner may test it with small size by hand — Amber still never places orders.
  If nothing clears it, say so plainly; the next options are roadmap D2
  (sub-minute data) or accepting Amber as a volatility indicator.
- **E. Ignition check** (2026-10-04, after the first ledger readings showed no
  edge and the owner ruled out "volatility indicator"): can a move be
  predicted from a CALM market (30/60-bar mid range < 0.5%, last 5 bars <
  0.1%, and candle high-low < 0.5% over every 20-bar slice), 1% within 15/30
  bars, entry the bar after? `amber/backtest/ignition.py`.
  Verdict routes the next step: `timing_signal` -> retarget the scanner to
  ignition (the product the owner described); `coin_choice_only` or
  `precursor_found` / `signal_matched_by` carried by coin-level features ->
  no timing information, treat as `no_precursor`;
  `no_precursor` -> 1m public bars cannot give early warning, D2 (order book,
  tick flow) is the only route; `underpowered` -> wait for more calm history.
  Since 2026-10-06 the arms are m30 (all/liquid/thin), m60 and h6 (6 h calm
  -> 3% in 4 h), with BTC/ETH lead and hour-scale features added.
  A verdict needs ≥20 independent moves in the test segment
  (`ignition.MIN_POSITIVE_EPISODES`); run with `--days 30` or more, because a
  1% move from calm is a ~1-in-1000-minutes event.
- **F. Live ignition channel** (2026-10-09, after `timing_signal:model`):
  shadow warnings + forward ledger, `scripts/run_ignition_report.py`, dashboard
  panel. `confirmed` -> turn on Telegram (`ignition.telegram: true`) with the
  owner; `not_confirmed` -> the 10-09 signal was a one-run artefact, back to
  D2; `underpowered` -> keep it running. **Confirmed 2026-10-17**; Telegram on.
- **G. Does the warning pay** (2026-10-17): the breakout bracket above, in the
  ignition ledger, shown in `run_ignition_report.py` and the dashboard.
  `profitable` -> the owner may trade it by hand at small size; Amber still
  places no orders. `not_shown_profitable` -> the warning is real but this
  way of trading it does not beat 15 bps; say so, do not re-tune the bracket on
  the same data (any new rule is a new forward test). **Result 2026-10-17:
  `not_shown_profitable`** (-16.1 bps a trade, lower bound -22.9, 102
  episodes, gross ≈ -1 bps). Open: the owner's choice between D2 (direction
  from sub-minute flow / order book), the 4H/1D pullback mode (historical
  backtest, fees small next to its targets), or keeping ignition as an
  information-only channel.
- **H. Pullback mode, 4H/1D** (proposed 2026-10-16, next in the queue): a
  read-only historical backtest on BTC/ETH (Bybit public klines, years):
  uptrend (EMA200) pullbacks — (a) a 3-5% drop within hours, (b) a retest of
  former resistance — x entries (limit at the zone, confirmation close,
  false-break reclaim), mirrored for downtrends; taker/maker fees, slippage
  and funding; control = random entries in the same trend state with the same
  stop/target; null fixture at the real arm count first. Result the same day
  the owner runs it; forward confirmation would take months (few trades).
  **Result 2026-10-17 (Binance archive): all 12 arms `not_profitable`;**
  long arms match the random-entry trend control. Not re-tuned.
- **I. External signal channel check** (approved 2026-10-17, last): collect
  t.me/ftcrypta's "long above X / short below Y" signals from the public web
  preview, replay on Bybit candles with exits fixed in advance and fees,
  against random level breakouts on the same coins and days. Count id gaps
  (deleted posts would bias it in his favour).
- **K. D2 data, collecting** (2026-10-17): order book + sub-minute flow
  (above). After ~3 weeks: a direction check built like the ignition check
  (calm bars / warnings, features from the book rows, within-coin ranks,
  exact family-wise bounds, null fixtures first) — the only route to
  direction left on public data.
- **J. Pump fade — backlog** (owner's idea 2026-10-17, parked by the owner):
  after an extreme pump (+30/50/100% in 24 h) short from the current level,
  expecting the dump; the first idea that supplies a direction. History test
  first (roadmap D12): whole universe incl. delisted, three entries, stop
  above the peak, half-pump target, ~3 days, real funding and small-cap
  slippage, random-short control, judged on the bounded mean (squeezes).
