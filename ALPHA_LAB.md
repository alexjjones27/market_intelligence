# alpha_lab — replicating "Automate Strategy Finding with LLM in Quant Investment"

A from-scratch, leakage-audited backtest of the three-stage framework in
Kou et al., *Automate Strategy Finding with LLM in Quant Investment*
(Findings of EMNLP 2025, pp. 18517–18533), adapted from SSE50/CSI300 to the
US S&P 500.

The goal is **a backtest worth trusting**, not a reproduction of the paper's
headline 53.17% cumulative return. Where the paper is silent, ambiguous, or
(in a few places) internally inconsistent, this repo says so in the code and in
the results rather than picking whichever reading produces a better number.

> **Status: Stage 1 complete and verified.** The Seed Alpha Factory, the data
> layer, and the leakage test suite are built and checked. Stages 2 (CSA/RPA
> agents) and 3 (MLP weight optimiser), plus the walk-forward engine and
> reporting, are scaffolded but not yet implemented.

---

## Layout

```
src/alpha_lab/
  config.py             # every knob the paper leaves open, loaded from YAML
  data/
    universe.py         # point-in-time S&P 500 membership
    prices.py           # daily OHLCV + split factors (Yahoo)
    fundamentals.py     # SEC EDGAR XBRL, keyed on actual filing dates
    macro.py            # macro proxies + benchmarks
    panel.py            # one aligned panel everything downstream reads
    cache.py, http.py   # on-disk cache, polite retry/backoff
  alphas/
    operators.py        # DELAY, SMA, EMA, RSI, ATR, ADX, ... (42 operators)
    library.py          # the 92 Appendix A.3 alphas, transcribed verbatim
    registry.py         # compute engine + per-alpha diagnostics
  agents/ models/ portfolio/ backtest/ reporting/   # Stages 2-3 (scaffolded)
configs/default.yaml    # k, n, wc/wr, IC horizon, dates, costs
scripts/
  build_universe.py     # fetch + cache membership snapshots
  build_panel.py        # fetch + cache the full market panel
  stage1_sanity_check.py# verification report and plots
tests/alphalab/         # operator, leakage, and split-basis tests
reports/                # generated output
```

Run order:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-alpha-lab.txt
.venv/bin/python scripts/build_panel.py           # ~15 min, cached thereafter
.venv/bin/python scripts/stage1_sanity_check.py   # writes reports/stage1_*
PYTHONPATH=src .venv/bin/python -m pytest tests/alphalab -q
```

Everything is config-driven: `configs/default.yaml` holds `top_k`, `drop_n`,
`w_confidence`/`w_risk`, the IC horizon, the walk-forward geometry, transaction
costs and the date ranges. Core logic never hard-codes them.

---

## Data, and what is honest about it

| Layer | Source | Point-in-time? |
|---|---|---|
| Index membership | Wikipedia **dated revisions** of *List of S&P 500 companies* | Yes — the page as it stood on each snapshot date |
| Daily OHLCV | Yahoo Finance chart endpoint | Yes |
| Splits | Yahoo split events | Yes |
| Fundamentals | SEC EDGAR XBRL `companyfacts` | Yes — keyed on `filed`, never on fiscal period end |
| Macro | Yahoo proxies (^TNX, ^IRX, ^VIX, DXY, oil, gold) | Yes |
| Benchmarks | ^GSPC (cap-weight), RSP (equal-weight) | Yes |

### Survivorship bias is reduced, not eliminated

Membership is reconstructed from **27 quarterly Wikipedia revisions**, which
yields **655 distinct tickers** that were S&P 500 members at some point between
2018 and 2024 — versus the 503 in the index today. The 152 names that left are
exactly the ones a naive "download today's constituents" study deletes.

The residual problem is that **Yahoo purges price history for delisted
symbols**. 110 of those 655 tickers (16.8%) return nothing, which is a real
hole and is quantified per snapshot in `reports/stage1_sanity_check.md` rather
than assumed away. The direction of the resulting bias is genuinely ambiguous:
acquisitions (ANSS, DFS, CTLT, ATVI — mostly at a premium) remove *winners*,
while failures (FRC, SIVB) remove *losers*.

### Fundamentals cannot be read early

Every XBRL fact carries both the period it describes and the date it was filed.
`fundamentals.py` keys exclusively on `filed`. Apple's FY2023 diluted EPS of
$6.13 enters the panel on **2023-11-03**, the 10-K filing date, and the panel
holds the prior $6.11 until then — not on 2023-09-30, the fiscal period end,
which is what a period-end join would do and is worth several percent a year on
any value or growth factor.

Flows are accumulated to trailing-twelve-month figures. Because most filers tag
Q1–Q3 quarterly but report only the full year in the 10-K, the untagged fiscal
Q4 is reconstructed as `annual − sum(three tagged quarters)` and becomes known
on the 10-K's filing date. Without that step the TTM window can never close on
contiguous quarters and every valuation factor refreshes annually instead of
quarterly.

### Splits: the bug this design catches

Yahoo restates prices for splits retroactively; SEC does not restate filings.
Divide a split-adjusted 2019 Apple price (~$50) by an as-filed FY2019 EPS
($11.89) and you get a 24% earnings yield — a P/E of 4 for a stock that traded
near 17. `prices.load_split_factors` builds `cum(t)/cum(T_end)` per ticker and
`fundamentals._apply_split_basis` applies it **at the filing date**: per-share
amounts are scaled one way, share counts the other, and market cap comes out
invariant. That invariance is asserted in `tests/alphalab/test_split_basis.py`.

---

## Stage 1 — the Seed Alpha Factory

The paper's Stage 1 has an LLM read 11 papers and emit ~100 formulaic alphas.
That step is unverifiable and is deliberately **not** replicated. Instead, the
Appendix A.3 table — the paper's own published output — is hard-coded as the
ground-truth factory: **92 alphas across 9 categories**, each carrying its A.3
short code verbatim in the source for line-by-line audit.

| Category | Count | Category | Count |
|---|---|---|---|
| momentum | 11 | quality | 8 |
| mean_reversion | 10 | growth | 10 |
| volatility | 10 | technical | 9 |
| fundamental | 6 | macro | 10 |
| liquidity | 18 | | |

### Guarantees

Two invariants are enforced by the test suite, not just by inspection:

1. **No look-ahead.** Every windowed operator is right-closed with
   `min_periods == window`; `ewm` uses `adjust=False`; `DELAY` raises on a
   negative shift. The headline test computes all 92 alphas on the full panel
   and again on a panel truncated at date *T*, and requires that values **and
   their non-null pattern** match on every date ≤ *T*. The NaN-pattern half
   matters: comparing only cells where both runs produced a value skips exactly
   the boundary window where a forward-looking operator reveals itself. A
   companion test injects a deliberate `shift(-5)` alpha and asserts the check
   catches it — a leakage test that cannot fail is worse than none.
2. **No cross-contamination.** Operators act column-wise; normalisation is
   same-day cross-sectional only. A separate test rewrites all future rows and
   asserts no past normalised value moves.

Alphas are computed on the **full** panel and masked to tradeable names
afterwards, so a stock's entry to or exit from the index cannot truncate its own
moving averages.

---

## What the replication found about the paper's design

These are properties of Appendix A.3 itself, not artefacts of our data access.
They matter because they bound what Stages 2–3 can possibly learn.

**1. The entire Macro category is cross-sectionally degenerate.**
`GDP − DELAY(GDP, n)` takes the same value for every stock on a given day. The
paper's own portfolio rule (Appendix A.8) *ranks stocks against each other*, so
a constant column changes no ranking, and its cross-sectional IC is not zero but
**undefined** — the Spearman correlation of a constant vector is NaN. One of the
paper's nine categories cannot contribute to the paper's own strategy. This
would hold identically with a full FRED feed.

**2. Roughly a quarter of the alphas are not scale-free.**
22 of 92 formulas carry units. `SMA(CLOSE, 20)` and `EMA(CLOSE, 20)` correlate
**+0.999** with log price across the cross-section — ranked against each other
they order stocks by nominal share price and essentially nothing else. `ATR(14)`
reaches +0.94, `STD(CLOSE, 20)` +0.85. As cross-sectional stock-ranking signals
these are largely price-level and size proxies.

**3. Seven alphas are duplicates.**
`ROC` and `Momentum Oscillator` are the *same expression*. `Bollinger Bands` and
`Percent B` differ by a factor of 100 — an identical ranking. `MEAN` and `SMA`
are the same operator, making `Mean Reversion` and `MA Reversion` identical.
`ATR(14)`, gross margin and debt-to-equity each appear in two categories. This
directly undercuts Algorithm 1's category-diversification claim: picking the
best alpha from each of two categories can pick the same signal twice.

**4. The paper's reported metrics are mutually inconsistent.**
Table 4 reports Sharpe 0.287 alongside 53.17% cumulative return and 0.762%
volatility — internally consistent only if that Sharpe is a *daily* figure
(≈4.5 annualised). But Table 7 reports Sharpe 1.94 for the same full model,
Table 9 reports 11.39 "overall", and Table 10 reports 13.33. These cannot all be
the same quantity. Table 7's text also cites "Sharpe Ratio of 1.73" against its
own table's 1.94, and Table 3 describes a combination IC of **−0.0587** as
"quite high" while reporting that removing one alpha moves it to **+0.0491**,
a sign flip described as a drop.

**5. Costs are not modelled at all, at ~38% daily turnover.**
Appendix A.8 states the k=13/n=5 configuration turns over ~38% of the portfolio
daily. At 10bps round-trip that is roughly 0.076%/day, on the order of **19% a
year** of drag — against a claimed 53% return. This repo reports every result
both gross and net.

**6. The paper's own selection prompt looks contaminated.**
Appendix A.7 instructs the LLM: *"When provided with the test set (performance
of the first 4 quarters of 2023): select the factors that will perform best in
the last quarter."* Selecting factors using test-period factor performance is
look-ahead. Separately, GPT-4o's training data covers the 2023 test window, so
asking it which factors worked in 2023 is not a clean out-of-sample question.
Our Stage 2 replaces this with a mechanical rolling-IC rule that can only see
data available as of each scoring date.

---

## Stage 1 verification results

From `reports/stage1_sanity_check.md` (regenerate with the script):

**Structural** — RSI within [0,100]; stochastic within [0,100]; Williams %R
within [−100,0]; historical volatility non-negative; per-day z-scores have mean
≈0 and std 1.000; no infinities anywhere.

**Behavioural, against price action a reader can check** —

| Check | Jan–Feb 2020 | 16–24 Mar 2020 |
|---|---|---|
| Universe-median annualised volatility | 17.7% | **98.9%** (5.6×) |
| Universe-median RSI(14) | 55.6 | **31.1** |
| Universe-median Bollinger position | — | **0.10** (pinned to lower band) |

Worst universe-wide Ulcer Index falls on **2020-03-25**, the COVID bottom week.

**Point-in-time fundamentals** — TTM EPS for AAPL, MSFT, JNJ and XOM changes
19–22 times over 2019–2024 with a **median gap of 90–91 days**, confirming the
values step on filing dates rather than drifting daily or updating at period end.

Plots: `reports/stage1_sanity_alphas.png` (alphas against AAPL price action) and
`reports/stage1_scale_dependence.png` (price-level contamination by alpha).

---

## Deliberate deviations from the paper

| Paper | Here | Why |
|---|---|---|
| SSE50 / CSI300 | S&P 500 | Chinese market data is not accessible; the user's stated target is US equities |
| LLM generates alphas from 11 papers | A.3 table hard-coded | The generation step is unverifiable; A.3 is the paper's own published output |
| LLM selects alphas from multimodal input incl. test-period data | Mechanical rolling out-of-sample IC | A.7's prompt is look-ahead, and GPT-4o has seen the test window |
| Multimodal (audio, video, images) | Numeric + fundamentals only | Not reproducible, and not the testable part of the claim |
| No transaction costs | 5bps cost + 5bps slippage per side, reported both ways | ~38% daily turnover makes costs first-order |
| Same-bar execution (implied) | `execution_lag_days: 1`, configurable | Signals from the close of *t* cannot be executed at the close of *t* |
| Single Jan 2023–Jan 2024 window | Rolling walk-forward across all folds | One window is not evidence |
| `CLOSE / BOOK_VALUE` | `CLOSE / book value **per share**` | The formula as written mixes per-share and company-level quantities |
| `DELAY(EPS, 1)` in a daily panel | 252 trading days (YoY on TTM) | A one-*day* delay on a quarterly series is identically zero |
| `STD(EPS, 5)` | 315 trading days (5 quarters) | Five daily bars of a quarterly series have zero variance |

Unspecified parameters we had to choose, all documented at the call site and
configurable: Bollinger `k=2.0`; Keltner centre `EMA(20)` with `2×ATR(10)`;
Yang-Zhang `VAR` window 20 days; macro `n=21` trading days.

---

## Known limitations

- **110 delisted tickers have no price history** (see above). This is the
  largest remaining threat to validity.
- **`bid_ask_spread` cannot be computed** — it needs quote-level NBBO data.
  Excluded rather than approximated by a high-low proxy that would double-count
  the `high_low_spread` factor already in the same category.
- **Nine of ten A.3 macro factors need FRED**, which is unreachable from this
  environment; market proxies stand in for six. They are cross-sectionally
  degenerate either way.
- **VWAP is approximated** by `(H+L+C)/3`; daily bars carry no true VWAP.
- **Restatements**: the first-filed value for a period is used, so the panel
  holds what the market had, not what the figure was later revised to.
- **Wikipedia membership lags real index changes** by days to weeks, and
  quarterly snapshots miss an add-then-drop inside one quarter.
- **Fundamental coverage varies by field** — 97% of stock-days for total assets,
  57% for gross profit (financials do not report it). Per-field coverage is in
  `reports/stage1_alpha_diagnostics.csv`.
