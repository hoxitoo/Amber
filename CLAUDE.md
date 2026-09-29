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
`evaluate_model` measured `pump`; permutation importance measured `pump`
(fixed 2026-09-29); the backtest still trades direction. Each reported numbers
about a head that decides nothing, and I reported them as if they meant
something.

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
  bound (`_episodes`, `_wilson_low`) with a family-wise correction
  (`family_z(n_compared)`). Alerts within one horizon of each other, on any
  symbol, are one observation.
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

The owner runs them verbatim on the VPS as root.
- Always from the project dir, as the service user, with the venv:
  `cd /opt/amber && sudo -u amber git pull`
  `cd /opt/amber && sudo -u amber /opt/amber/.venv/bin/python scripts/<x>.py`
  Plain `git pull` as root fails on ownership; plain `python` has no deps; a
  relative path from `/root` does not exist.
- Say which services need a restart and why.
- Say how long to wait before a result means anything (the training window is
  `max_candles_per_symbol` long; a new feature carries zeros until it rolls over).
- Give the exact lines to look for in the output, and what each outcome means.

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
- As of 2026-09-29 the move model is statistically indistinguishable from the
  one-feature rule `range_atr_14` (baseline check; details in section 10). Everything available on 1m
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
  "Журнал сделок". Started on first run at the end of the existing logs. Per alert: ts,
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
- **D. Decide** at ≥30 independent episodes (rule 2 bounds, family-wise over
  the rules compared): if a rule's net result has a lower bound above 0, the
  owner may test it with small size by hand — Amber still never places orders.
  If nothing clears it, say so plainly; the next options are roadmap D2
  (sub-minute data) or accepting Amber as a volatility indicator.
