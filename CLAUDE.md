# Rules for working on Amber

Read this first, every session. Each rule exists because breaking it already
cost this project a wrong conclusion, a silent failure, or a round trip with the
owner. The incident is named so the rule is not mistaken for general advice.

The owner runs this 24/7 and acts on what I report. A confident wrong answer is
worse than "I don't know yet".

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
  `run_operating_curve.py`, `run_baseline_check.py`. All read-only.
- As of 2026-09-29 the move model is statistically indistinguishable from the
  one-feature rule `range_atr_14` (baseline check). Everything available on 1m
  bars says "a move is already under way"; see roadmap D2/D11.
- `docs/target_review_2026-09.md` holds the reasoning behind the current target.
