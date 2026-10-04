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
  predicted from a CALM market (30/60-bar range < 0.5% and last 5 bars <
  0.1%), 1% within 15/30 bars, entry the bar after? `amber/backtest/ignition.py`.
  Verdict routes the next step: `precursor_found` / `signal_matched_by` ->
  retarget the scanner to ignition (the product the owner described);
  `no_precursor` -> 1m public bars cannot give early warning, D2 (order book,
  tick flow) is the only route; `underpowered` -> wait for more calm history.
  A verdict needs ≥20 independent moves in the test segment
  (`ignition.MIN_POSITIVE_EPISODES`); run with `--days 30` or more, because a
  1% move from calm is a ~1-in-1000-minutes event.
