# Target definition review — September 2026

After five days on the updated pipeline the dashboard showed a win rate of
0.625 against a break-even of 0.566, and a profit factor of 0.74. Those cannot
both be good news, and working out why produced the findings below.

## 1. Win rate and profit factor were measured over different denominators

`_aggregate` in `amber/backtest/backtester.py` computes

    win_rate      = TP / (TP + SL)          # timeouts excluded
    profit_factor = gross_profit / gross_loss   # every trade, timeout pays cost

A timeout appears in neither the wins nor the losses of the win-rate ratio, but
pays the full round trip in the profit factor. Solving the pair for the 197
recorded trades:

| quantity | value |
|---|---|
| resolution rate | 30.2% |
| TP / SL / timeout | 37 / 22 / 138 |
| expectancy | −3.89 bps per trade |
| share of losses that are timeout costs | 42% |

For PF > 1 at the same directional quality the resolution rate would have to
reach ~54%. The dashboard now shows the resolution rate, the TP/SL/timeout
split and expectancy alongside the win rate, and warns when timeouts dominate.

## 2. The win rate itself is not yet distinguishable from break-even

0.625 was measured over 59 resolved trades. The 95% interval is [0.50, 0.75],
which contains both the 0.566 break-even and the previous 0.496 reading. The
apparent improvement is within noise; separating it from break-even needs on
the order of 260 resolved trades.

## 3. The edge does not survive the delay between alert and action

| measurement | value | lift over base |
|---|---|---|
| base rate (pump) | 20.4% | — |
| precision at threshold, scoring bar | 36.8% | ×1.80 |
| precision in backtest, entry lagged one bar | 18.8% | ≈×0.9–1.1 |

The comparison is not perfectly like-for-like — the backtest pools pump and
dump, whose base rates differ, and applies concurrency gating — but the drop is
too large to attribute to that. This is the finding that matters most for a
tool whose output is an alert a human acts on after the bar closes, and no
barrier setting addresses it.

## 4. The barrier cancels the model's strongest signal

Permutation importance puts `range_atr_14` at 32.4% of the total, with the top
five features at 97% and 9 of 21 carrying nothing. The model is largely a
volatility forecaster. The production barrier is `k · σ_fast · √h`, scaled by
the *same* fast volatility — so a correctly detected volatility spike widens
the target in proportion and cancels itself.

The fix is not to remove volatility scaling, which is what keeps a signal on
BTC comparable to one on a small cap. It is to scale the barrier by a *slow*
baseline (24h realised volatility) so that "fast volatility is rising against
its own baseline" becomes informative rather than self-cancelling.

## 5. Two PSI numbers were answering different questions

The model tab pooled the universe against the model's `train_reference` and
read `low`. The data tab scored each symbol against that same pooled reference
and read `high`, with max PSI 7.2–12.4, for all 27 symbols for five days.

12.434 is the arithmetic signature of all mass landing in one bin of a ten-bin
grid with uniform expected shares. A quiet coin occupies one decile of a
universe-wide grid, so every one of its rows falls in that bin — cross-sectional
spread, not drift. Per-symbol drift now references each symbol's own earlier
window; the pooled comparison remains on the model tab.

## How the sweep decides

`scripts/run_label_sweep.py` trains one model per arm over
horizon × barrier ruler × label shape and reports precision at a fixed alert
budget, at lag 0 and lag 1. It writes only `logs/label_sweep.json`.

Ranking is by `lift1_lo` — the lower bound on lift at lag 1 — never by the
point estimate. The sweep was validated against a 27-symbol pure random walk,
where the correct answer is known to be "no edge", and failed that null twice
before passing:

1. Ranking on the point estimate returned a top arm at **lift 1.78**, precision
   measured on 20 alerts. Fixed by reporting and ranking on a Wilson lower
   bound.
2. With per-arm 95% bounds it still returned **`edge_survives_lag`**: comparing
   24 arms is 24 chances to cross a 95% bound, and something usually does.
   Fixed by `family_z`, a Bonferroni correction that splits alpha across the
   family, so the bound answers "better than random *after looking 24 times*".

A tool built to keep the project off noise must not itself produce noise, so
the null case is a test at the full 24-arm grid
(`tests/test_label_sweep.py::test_random_walk_yields_no_edge`). An earlier
version of that test used 8 arms and passed while the real 24-arm run failed —
the arm count is part of what is being tested.

Two guards on the reading:

- `floored%` — when `threshold_floor` (0.5%) rather than the ruler sets the
  barrier, a "volatility-scaled" arm is really a fixed one and the ruler axis
  measures nothing.
- `underpowered` — below ~60 alerts per arm on the test segment every interval
  is wide enough to swallow the differences between arms.

Outcomes:

- `edge_survives_lag` — significant on test and reproduced on calib. Adopt that
  arm's labelling.
- `test_only` — significant on test but not on calib. Treat as noise.
- `inconclusive` — the best point estimate looks interesting but its interval
  contains 1.0. Widen `--budget` or `--max-candles` and rerun. This is what the
  random-walk null returns, so it is not by itself evidence of anything.
- `no_edge` — across every arm the model is indistinguishable from random at
  lag 1. The target is not the problem; the feature set is, and the next step
  is a new information source (order book depth, liquidations, OI) rather than
  further tuning.

Reference points from the validation run, for reading a real one against:

| run | best `lift1` | best `lift1_lo` | verdict |
|---|---|---|---|
| 27-symbol random walk, no edge, 117 alerts/arm | 1.28 | 0.89 | `inconclusive` |
