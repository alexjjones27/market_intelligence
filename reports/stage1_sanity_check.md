# Stage 1 - Seed Alpha Factory: sanity check

- Panel: **1633 sessions** (2018-01-02 to 2024-06-28) x **545 tickers**
- Alphas registered: **92** across **9** categories
- Alphas computed: **87**; failures: 0
- Mean tradeable names/day: **452**

## 1. Structural checks

| Check | Result | Detail |
|---|---|---|
| RSI stays within [0, 100] | PASS | observed [3.20, 94.20] |
| Stochastic oscillator within [0, 100] | PASS | observed [0.00, 100.00] |
| Williams %R within [-100, 0] | PASS | observed [-100.00, -0.00] |
| Historical volatility non-negative | PASS | min 0.0064 |
| Cross-sectional z-scores standardised per day | PASS | mean -5.67e-18, std 1.0000 |
| No infinities anywhere in normalised alphas | PASS |  |

## 2. Behavioural checks against known price action

- **Volatility spikes in the COVID crash.** Universe-median annualised historical volatility: **17.7%** in Jan-Feb 2020 vs **98.9%** on 16-24 Mar 2020 (5.6x). PASS
- **RSI collapses at the crash low.** Universe-median RSI(14): **55.6** in Jan-Feb 2020 vs **31.1** on 16-24 Mar 2020. PASS
- **Bollinger position pinned to the lower band in the crash.** Universe-median (CLOSE-LOWER)/(UPPER-LOWER) = **0.10** on 16-24 Mar 2020 (0 = lower band, 1 = upper). PASS
- **Worst universe-wide Ulcer Index** (drawdown pain) falls on **2020-03-25** -- the COVID crash bottom week. PASS

## 3. Fundamentals move only on filing dates

| Ticker | Distinct TTM EPS values 2019-2024 | Change dates | Median gap (days) |
|---|---|---|---|
| AAPL | 22 | 21 | 91 |
| MSFT | 23 | 22 | 91 |
| JNJ | 20 | 19 | 90 |
| XOM | 23 | 22 | 91 |

Roughly-quarterly gaps confirm the values step on SEC filing dates rather than drifting daily or updating at fiscal period end.

## 4. Diagnostics worth knowing before Stage 2

### Cross-sectionally degenerate: 6 alphas

These take the same value for every stock on a given day, so they cannot rank anything and their cross-sectional IC is undefined (not zero -- undefined; the Spearman correlation of a constant vector is NaN).

| Alpha | Category |
|---|---|
| `dollar_index_change` | macro |
| `gold_price_change` | macro |
| `implied_vol_change` | macro |
| `interest_rate` | macro |
| `oil_price_change` | macro |
| `short_rate` | macro |

### Not computable from available data: 5 alphas

| Alpha | Reason |
|---|---|
| `bid_ask_spread` | Needs quote-level (NBBO) data. Daily OHLCV cannot supply it, and no free intraday quote source is reachable here, so it is excluded rather than approximated by a high-low proxy that would double-count the high_low_spread factor already in this category. |
| `gdp_growth` | FRED GDPC1 unreachable from this environment. Would in any case be cross-sectionally constant. |
| `unemployment_rate` | FRED UNRATE unreachable. Would in any case be cross-sectionally constant. |
| `retail_sales_growth` | FRED RSAFS unreachable. Would in any case be cross-sectionally constant. |
| `housing_starts_growth` | FRED HOUST unreachable. Would in any case be cross-sectionally constant. |

### Duplicate formulas in Appendix A.3: 7 alphas

A.3 lists these twice, sometimes in different categories. That matters for a *category-diversified* selection rule: picking one from each of two categories can pick the same signal twice.

| Alpha | Duplicate of | Category |
|---|---|---|
| `momentum_oscillator` | `roc` | momentum |
| `ma_reversion` | `mean_reversion` | mean_reversion |
| `percent_b` | `bollinger_position` | mean_reversion |
| `debt_to_equity_liq` | `debt_to_equity` | liquidity |
| `gross_profit_margin_liq` | `gross_profit_margin` | liquidity |
| `sales_growth` | `revenue_growth` | growth |
| `atr_technical` | `atr` | technical |

### Price-level contamination: 11 alphas correlate |rho| > 0.5 with log price

Many A.3 formulas carry units. `CLOSE - DELAY(CLOSE, 14)` is a *dollar* change, so a \$500 stock outranks a \$20 stock arithmetically, not predictively. Ranked cross-sectionally these behave largely as price-level or size proxies. Mean daily Spearman rho against log(price):

| Alpha | Category | rho vs log price | scale-free? |
|---|---|---|---|
| `exp_moving_average` | technical | +0.999 | NO |
| `moving_average` | technical | +0.999 | NO |
| `atr` | volatility | +0.943 | NO |
| `atr_technical` | technical | +0.943 | NO |
| `liquidity_ratio` | liquidity | -0.859 | yes |
| `std_dev` | volatility | +0.848 | NO |
| `bollinger_bands` | technical | +0.848 | NO |
| `distance_from_low` | mean_reversion | +0.741 | NO |
| `avg_trading_volume` | liquidity | -0.601 | NO |
| `distance_from_high` | mean_reversion | +0.593 | NO |
| `trading_volume` | liquidity | -0.581 | NO |
| `price_to_cash_flow` | liquidity | +0.463 | yes |

### Thin coverage: 1 alphas valid on <50% of tradeable stock-days

| Alpha | Category | % of tradeable stock-days with a value |
|---|---|---|
| `cash_conversion_cycle` | quality | 32.4% |

## 5. Survivorship: how much of the point-in-time universe we can price

- Distinct tickers that were ever S&P 500 members in the window: **505** per snapshot, **655** distinct overall
- Tickers with no price history at all: **110** (16.8%)
- Mean share of index members missing prices on a given snapshot: **10.9%**

Yahoo purges history for delisted symbols, so names that left the index through bankruptcy, acquisition or a ticker change return nothing. This is a *residual* survivorship bias on top of an otherwise point-in-time universe, and it is quantified here rather than assumed away. Direction is genuinely ambiguous: acquisitions (mostly at a premium) remove winners, while failures like FRC remove losers.

Missing tickers (sample): `ABC`, `ABMD`, `ADS`, `AGN`, `ALXN`, `ANSS`, `ANTM`, `APC`, `ARNC`, `ATVI`, `AVB`, `BCR`, `BHGE`, `BK`, `BLL`, `CA`, `CBG`, `CBS`, `CDAY`, `CELG`, `CERN`, `CHK`, `CMA`, `COG`, `CSRA`, `CTL`, `CTLT`, `CTRA`, `CTXS`, `CXO`, `DAY`, `DFS`, `DISCA`, `DISCK`, `DISH`, `DPS`, `DRE`, `DWDP`, `EA`, `EQR`
