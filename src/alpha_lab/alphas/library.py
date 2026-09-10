"""The Seed Alpha Factory: a literal transcription of Appendix A.3.

Every alpha below carries its paper formula verbatim in ``formula=``. Where the
paper's notation is ambiguous or dimensionally inconsistent, the deviation is
recorded in ``notes=`` rather than being quietly "fixed" -- the point of this
build is to test the paper's design, not an improved version of it.

The recurring ambiguities, and how each is resolved:

* **Per-share vs. company-level.** A.3 writes ``P/B = CLOSE / BOOK_VALUE`` and
  ``Sales-to-Price = SALES / CLOSE``. ``CLOSE`` is a per-share price while book
  value and sales are company-level totals, so those ratios as written are
  dimensionally meaningless (they scale with share count). We use the per-share
  figure, which is what the ratio's name means everywhere in finance.
* **``DELAY`` on fundamentals.** See ``AlphaConfig.fundamental_growth_lag_days``.
* **Unspecified windows.** Bollinger k, Keltner's centre line and multiplier,
  and Yang-Zhang's VAR horizon are not given in A.3; conventional values are
  used and named in the notes.
* **Scale-freeness.** Many A.3 factors carry units -- ``CLOSE - DELAY(CLOSE, 14)``
  is a dollar change, ``VOLUME * CLOSE`` is dollars traded. Ranked across a
  cross-section these largely order stocks by price level or company size
  rather than by any predictive pattern. Each such alpha is marked
  ``scale_free=False`` and the diagnostics report its rank correlation with log
  price, so the effect is visible instead of assumed.
"""
from __future__ import annotations

from alpha_lab.alphas import operators as ops
from alpha_lab.alphas.operators import (
    ABS, ADX, ATR, BOLLINGER, CCI, DELAY, EMA, KELTNER, MAX, MEAN, MIN, OBV,
    RSI, SIGN, SMA, STD, SUM, TYPICAL_PRICE, WILLIAMS_R, safe_divide,
)
from alpha_lab.alphas.registry import AlphaContext, AlphaRegistry, AlphaSpec

registry = AlphaRegistry()
_add = registry.add


def _spec(name, category, formula, fn, **kw):
    _add(AlphaSpec(name=name, category=category, formula=formula, fn=fn, **kw))


# ============================================================ 1. MOMENTUM (11)

_spec("price_momentum", "momentum", "(CLOSE - DELAY(CLOSE, 14))",
      lambda c: c.close - DELAY(c.close, 14),
      scale_free=False,
      notes="Dollar change, not a return: a $500 stock outranks a $20 stock for "
            "arithmetic reasons. Compare with momentum_oscillator, its scale-free twin.")

_spec("volume_momentum", "momentum", "(VOLUME - DELAY(VOLUME, 14))",
      lambda c: c.volume - DELAY(c.volume, 14),
      scale_free=False, notes="Share-count change; dominated by large-float names.")

_spec("rsi_momentum", "momentum", "(RSI - DELAY(RSI, 14))",
      lambda c: RSI(c.close, 14) - DELAY(RSI(c.close, 14), 14))

_spec("roc", "momentum", "((CLOSE / DELAY(CLOSE, 14)) - 1)",
      lambda c: safe_divide(c.close, DELAY(c.close, 14)) - 1.0,
      notes="Algebraically identical to momentum_oscillator below; A.3 lists both.")

_spec("macd_momentum", "momentum", "(MACD - DELAY(MACD, 14))",
      lambda c: ops.MACD(c.close) - DELAY(ops.MACD(c.close), 14),
      scale_free=False, notes="MACD is an EMA difference in dollars, so it scales with price.")

_spec("momentum_oscillator", "momentum", "((CLOSE - DELAY(CLOSE, 14)) / DELAY(CLOSE, 14))",
      lambda c: safe_divide(c.close - DELAY(c.close, 14), DELAY(c.close, 14)),
      duplicate_of="roc",
      notes="Identical expression to roc. Kept so the category count matches A.3, "
            "but a 'diversified' selection picking both is picking one factor twice.")


def _cmo(c: AlphaContext, n: int = 14):
    delta = c.close.diff()
    gains = SUM(delta.clip(lower=0.0), n)
    losses = SUM((-delta).clip(lower=0.0), n)
    return 100.0 * safe_divide(gains - losses, gains + losses)


_spec("cmo", "momentum",
      "(SUM(IF(CLOSE-DELAY(CLOSE,1)>0, CLOSE-DELAY(CLOSE,1), 0), 14) - SUM(IF(...<0, ...,0),14)) "
      "/ (SUM(...)+SUM(...)) * 100",
      _cmo)

_spec("smi", "momentum",
      "((CLOSE - MIN(LOW,14)) - (MAX(HIGH,14) - CLOSE)) / (MAX(HIGH,14) - MIN(LOW,14))",
      lambda c: safe_divide(
          (c.close - MIN(c.low, 14)) - (MAX(c.high, 14) - c.close),
          MAX(c.high, 14) - MIN(c.low, 14)))

_spec("atr_momentum", "momentum", "(ATR - DELAY(ATR, 14))",
      lambda c: ATR(c.high, c.low, c.close, 14) - DELAY(ATR(c.high, c.low, c.close, 14), 14),
      scale_free=False)

_spec("dpo", "momentum", "(CLOSE - DELAY(SMA(CLOSE, 14), 7))",
      lambda c: c.close - DELAY(SMA(c.close, 14), 7),
      scale_free=False,
      notes="The worked example in A.1/Figure 4. Dollar-denominated.")

_spec("adx_momentum", "momentum", "(ADX - DELAY(ADX, 14))",
      lambda c: ADX(c.high, c.low, c.close, 14) - DELAY(ADX(c.high, c.low, c.close, 14), 14))


# ====================================================== 2. MEAN REVERSION (10)

_spec("mean_reversion", "mean_reversion", "(MEAN(CLOSE, 20) - CLOSE)",
      lambda c: MEAN(c.close, 20) - c.close,
      scale_free=False,
      notes="MEAN and SMA are the same operator, so this is identical to ma_reversion.")

_spec("zscore_reversion", "mean_reversion", "(CLOSE - MEAN(CLOSE, 20)) / STD(CLOSE, 20)",
      lambda c: safe_divide(c.close - MEAN(c.close, 20), STD(c.close, 20)))

_spec("bollinger_position", "mean_reversion",
      "(CLOSE - LOWER_BAND) / (UPPER_BAND - LOWER_BAND)",
      lambda c: _boll_pos(c),
      notes="k=2.0 on a 20-day SMA (A.3 does not state the band width).")

_spec("keltner_position", "mean_reversion",
      "(CLOSE - LOWER_CHANNEL) / (UPPER_CHANNEL - LOWER_CHANNEL)",
      lambda c: _keltner_pos(c),
      notes="EMA(20) centre with 2x ATR(10) bands; A.3 specifies neither.")

_spec("ma_reversion", "mean_reversion", "(SMA(CLOSE, 20) - CLOSE)",
      lambda c: SMA(c.close, 20) - c.close,
      duplicate_of="mean_reversion", scale_free=False)

_spec("ema_reversion", "mean_reversion", "(EMA(CLOSE, 20) - CLOSE)",
      lambda c: EMA(c.close, 20) - c.close, scale_free=False)

_spec("distance_from_high", "mean_reversion", "(MAX(HIGH, 20) - CLOSE)",
      lambda c: MAX(c.high, 20) - c.close, scale_free=False)

_spec("distance_from_low", "mean_reversion", "(CLOSE - MIN(LOW, 20))",
      lambda c: c.close - MIN(c.low, 20), scale_free=False)

_spec("rsi_reversion", "mean_reversion", "(100 - RSI)",
      lambda c: 100.0 - RSI(c.close, 14))

_spec("percent_b", "mean_reversion",
      "((CLOSE - LOWER_BAND) / (UPPER_BAND - LOWER_BAND)) * 100",
      lambda c: _boll_pos(c) * 100.0,
      duplicate_of="bollinger_position",
      notes="bollinger_position times 100 -- an identical ranking, listed twice by A.3.")


def _boll_pos(c: AlphaContext):
    upper, lower = BOLLINGER(c.close, 20, 2.0)
    return safe_divide(c.close - lower, upper - lower)


def _keltner_pos(c: AlphaContext):
    upper, lower = KELTNER(c.high, c.low, c.close, 20, 10, 2.0)
    return safe_divide(c.close - lower, upper - lower)


# ========================================================== 3. VOLATILITY (10)

_spec("std_dev", "volatility", "STD(CLOSE, 20)",
      lambda c: STD(c.close, 20), scale_free=False,
      notes="Dollar standard deviation; higher-priced stocks rank higher mechanically.")

_spec("atr", "volatility", "ATR(14)",
      lambda c: ATR(c.high, c.low, c.close, 14), scale_free=False)

_spec("bollinger_width", "volatility", "(UPPER_BAND - LOWER_BAND) / SMA(CLOSE, 20)",
      lambda c: _boll_width(c))

_spec("historical_volatility", "volatility", "STD(RETURNS, 20) * SQRT(252)",
      lambda c: STD(c.returns, 20) * (252 ** 0.5))

_spec("volatility_ratio", "volatility", "STD(CLOSE, 10) / STD(CLOSE, 50)",
      lambda c: safe_divide(STD(c.close, 10), STD(c.close, 50)))

_spec("chaikin_volatility", "volatility",
      "(EMA(HIGH - LOW, 10) / DELAY(EMA(HIGH - LOW, 10), 10)) - 1",
      lambda c: safe_divide(EMA(c.high - c.low, 10), DELAY(EMA(c.high - c.low, 10), 10)) - 1.0)

_spec("garman_klass", "volatility",
      "SQRT(0.5*LOG(HIGH/LOW)^2 - (2*LOG(2)-1)*LOG(CLOSE/OPEN)^2)",
      lambda c: ops.GARMAN_KLASS(c.open, c.high, c.low, c.close),
      notes="A.3 gives no window, so this is a single-bar estimator and is "
            "correspondingly noisy; convention averages the variance over ~20 days first.")

_spec("parkinson", "volatility", "SQRT((1/(4*N*LOG(2))) * SUM(LOG(HIGH/LOW)^2, 20))",
      lambda c: ops.PARKINSON(c.high, c.low, 20))

_spec("yang_zhang", "volatility",
      "SQRT(VAR(LOG(CLOSE/OPEN)) + 0.5*VAR(LOG(HIGH/OPEN)-LOG(LOW/OPEN)) "
      "+ 0.25*VAR(LOG(CLOSE/DELAY(OPEN,1))))",
      lambda c: ops.YANG_ZHANG(c.open, c.high, c.low, c.close, 20),
      notes="Transcribed as written; this is NOT the canonical Yang-Zhang estimator "
            "(no k-weighting, no Rogers-Satchell term). VAR window set to 20 days.")

_spec("ulcer_index", "volatility", "SQRT(MEAN(DRAWDOWN^2, 14))",
      lambda c: ops.ULCER_INDEX(c.close, 14))


def _boll_width(c: AlphaContext):
    upper, lower = BOLLINGER(c.close, 20, 2.0)
    return safe_divide(upper - lower, SMA(c.close, 20))


# ========================================================= 4. FUNDAMENTAL (6)

_spec("price_to_earnings", "fundamental", "(CLOSE / EPS)",
      lambda c: safe_divide(c.close, c.f("eps")),
      notes="TTM diluted EPS, known from its SEC filing date. Undefined/negative for "
            "loss-makers, which is why earnings_yield is the more usable form.")

_spec("price_to_book", "fundamental", "(CLOSE / BOOK_VALUE)",
      lambda c: safe_divide(c.close, c.f("book_value_per_share")),
      notes="Uses book value PER SHARE: A.3's CLOSE/BOOK_VALUE mixes a per-share "
            "price with a company-level total and is not a P/B ratio as written.")

_spec("dividend_yield", "fundamental", "(DIVIDENDS / CLOSE)",
      lambda c: safe_divide(c.f("dividends_per_share"), c.close),
      notes="TTM declared dividends per share.")

_spec("earnings_yield", "fundamental", "(EPS / CLOSE)",
      lambda c: safe_divide(c.f("eps"), c.close),
      notes="Reciprocal of price_to_earnings, but better behaved: it stays finite "
            "and correctly ordered through zero earnings.")

_spec("sales_to_price", "fundamental", "(SALES / CLOSE)",
      lambda c: safe_divide(c.f("revenue_per_share"), c.close),
      notes="Revenue per share over price (see price_to_book note on units).")

_spec("cash_flow_yield", "fundamental", "(OPERATING_CASH_FLOW / CLOSE)",
      lambda c: safe_divide(c.f("cash_flow_per_share"), c.close),
      notes="Operating cash flow per share over price.")


# =========================================================== 5. LIQUIDITY (18)
# A.3's Liquidity block is a grab-bag: it also contains leverage, profitability
# and valuation ratios, and repeats High-Low Spread and Dollar Volume twice.
# Transcribed as printed, with the repeats marked rather than dropped.

_spec("trading_volume", "liquidity", "VOLUME", lambda c: c.volume, scale_free=False)

_spec("avg_trading_volume", "liquidity", "MEAN(VOLUME, 20)",
      lambda c: MEAN(c.volume, 20), scale_free=False)

_spec("vroc", "liquidity", "(VOLUME - DELAY(VOLUME, 14)) / DELAY(VOLUME, 14)",
      lambda c: safe_divide(c.volume - DELAY(c.volume, 14), DELAY(c.volume, 14)))

_spec("obv", "liquidity", "SUM(VOLUME * SIGN(CLOSE - DELAY(CLOSE, 1)))",
      lambda c: OBV(c.close, c.volume), scale_free=False,
      notes="Unbounded cumulative sum with no window, exactly as A.3 writes it; "
            "its level depends on how long the name has been listed.")

_spec("liquidity_ratio", "liquidity", "VOLUME / MARKET_CAP",
      lambda c: safe_divide(c.volume, c.f("market_cap")))

_spec("turnover_rate", "liquidity", "VOLUME / SHARES_OUTSTANDING",
      lambda c: safe_divide(c.volume, c.f("shares_outstanding")))

_spec("amihud_illiquidity", "liquidity", "ABS(RETURN) / VOLUME",
      lambda c: safe_divide(ABS(c.returns), c.volume), scale_free=False,
      notes="A.3 divides by share volume; Amihud's own measure uses dollar volume.")

_spec("high_low_spread", "liquidity", "(HIGH - LOW) / CLOSE",
      lambda c: safe_divide(c.high - c.low, c.close))

_spec("dollar_volume", "liquidity", "VOLUME * CLOSE",
      lambda c: c.volume * c.close, scale_free=False,
      notes="A near-pure size proxy: ranks megacaps top every day.")

_spec("debt_to_equity_liq", "liquidity", "(TOTAL_DEBT / TOTAL_EQUITY)",
      lambda c: safe_divide(c.f("total_debt"), c.f("total_equity")),
      duplicate_of="debt_to_equity",
      notes="A.3 lists debt-to-equity under both Liquidity and Quality.")

_spec("return_on_equity", "liquidity", "(NET_INCOME / EQUITY)",
      lambda c: safe_divide(c.f("net_income"), c.f("total_equity")))

_spec("return_on_assets", "liquidity", "(NET_INCOME / TOTAL_ASSETS)",
      lambda c: safe_divide(c.f("net_income"), c.f("total_assets")))

_spec("gross_profit_margin_liq", "liquidity", "(GROSS_PROFIT / REVENUE)",
      lambda c: safe_divide(c.f("gross_profit"), c.f("revenue")),
      duplicate_of="gross_profit_margin",
      notes="Repeated verbatim in the Quality block.")

_spec("price_to_sales", "liquidity", "(CLOSE / SALES)",
      lambda c: safe_divide(c.close, c.f("revenue_per_share")))

_spec("price_to_cash_flow", "liquidity", "(CLOSE / OPERATING_CASH_FLOW)",
      lambda c: safe_divide(c.close, c.f("cash_flow_per_share")))

_spec("book_to_market", "liquidity", "(BOOK_VALUE / CLOSE)",
      lambda c: safe_divide(c.f("book_value_per_share"), c.close))

_spec("ev_to_ebitda", "liquidity", "(ENTERPRISE_VALUE / EBITDA)",
      lambda c: safe_divide(c.f("enterprise_value"), c.f("ebitda")))

_spec("bid_ask_spread", "liquidity", "(ASK_PRICE - BID_PRICE) / MID_PRICE",
      lambda c: c.panel.empty(), available=False,
      notes="Needs quote-level (NBBO) data. Daily OHLCV cannot supply it, and no "
            "free intraday quote source is reachable here, so it is excluded "
            "rather than approximated by a high-low proxy that would double-count "
            "the high_low_spread factor already in this category.")


# ============================================================= 6. QUALITY (8)

_spec("gross_profit_margin", "quality", "(GROSS_PROFIT / REVENUE)",
      lambda c: safe_divide(c.f("gross_profit"), c.f("revenue")))

_spec("operating_profit_margin", "quality", "(OPERATING_INCOME / REVENUE)",
      lambda c: safe_divide(c.f("operating_income"), c.f("revenue")))

_spec("net_profit_margin", "quality", "(NET_INCOME / REVENUE)",
      lambda c: safe_divide(c.f("net_income"), c.f("revenue")))

_spec("earnings_stability", "quality", "STD(EPS, 5) / MEAN(EPS, 5)",
      lambda c: _earnings_stability(c),
      notes="Window read as five reporting periods (315 trading days), not five "
            "days: five daily bars of a quarterly TTM series have zero variance.")

_spec("debt_to_equity", "quality", "(TOTAL_DEBT / TOTAL_EQUITY)",
      lambda c: safe_divide(c.f("total_debt"), c.f("total_equity")))

_spec("interest_coverage", "quality", "(EBIT / INTEREST_EXPENSE)",
      lambda c: safe_divide(c.f("ebit"), c.f("interest_expense")),
      notes="EBIT proxied by operating income. Interest expense is untagged by many "
            "filers, so coverage is thin.")

_spec("cash_conversion_cycle", "quality", "(DIO + DSO - DPO)",
      lambda c: _cash_conversion_cycle(c),
      notes="DIO=365*inventory/COGS, DSO=365*receivables/revenue, DPO=365*payables/COGS. "
            "Meaningless for financials, which carry no inventory.")

_spec("asset_turnover", "quality", "(REVENUE / TOTAL_ASSETS)",
      lambda c: safe_divide(c.f("revenue"), c.f("total_assets")))


def _earnings_stability(c: AlphaContext):
    n = c.cfg.alphas.earnings_stability_window
    eps = c.f("eps")
    return safe_divide(STD(eps, n), MEAN(eps, n))


def _cash_conversion_cycle(c: AlphaContext):
    cogs, revenue = c.f("cogs"), c.f("revenue")
    dio = 365.0 * safe_divide(c.f("inventory"), cogs)
    dso = 365.0 * safe_divide(c.f("receivables"), revenue)
    dpo = 365.0 * safe_divide(c.f("payables"), cogs)
    return dio + dso - dpo


# ============================================================== 7. GROWTH (10)


def _growth(field: str):
    def fn(c: AlphaContext):
        lag = c.cfg.alphas.fundamental_growth_lag_days
        x = c.f(field)
        return safe_divide(x, DELAY(x, lag)) - 1.0
    return fn


_GROWTH_FIELDS = [
    ("earnings_growth", "eps", "(EPS / DELAY(EPS, 1) - 1)"),
    ("revenue_growth", "revenue", "(REVENUE / DELAY(REVENUE, 1) - 1)"),
    ("ebitda_growth", "ebitda", "(EBITDA / DELAY(EBITDA, 1) - 1)"),
    ("cash_flow_growth", "operating_cash_flow", "(CASH_FLOW / DELAY(CASH_FLOW, 1) - 1)"),
    ("dividends_growth", "dividends_per_share", "(DIVIDENDS / DELAY(DIVIDENDS, 1) - 1)"),
    ("book_value_growth", "book_value", "(BOOK_VALUE / DELAY(BOOK_VALUE, 1) - 1)"),
    ("sales_growth", "revenue", "(SALES / DELAY(SALES, 1) - 1)"),
    ("asset_growth", "total_assets", "(ASSETS / DELAY(ASSETS, 1) - 1)"),
    ("equity_growth", "total_equity", "(EQUITY / DELAY(EQUITY, 1) - 1)"),
    ("retained_earnings_growth", "retained_earnings",
     "(RETAINED_EARNINGS / DELAY(RETAINED_EARNINGS, 1) - 1)"),
]

for _name, _field, _formula in _GROWTH_FIELDS:
    _spec(_name, "growth", _formula, _growth(_field),
          duplicate_of="revenue_growth" if _name == "sales_growth" else None,
          notes=("A.3 lists SALES and REVENUE growth separately; both resolve to the "
                 "same XBRL revenue concept." if _name == "sales_growth" else
                 "DELAY(X,1) read as one year (252 trading days) of the TTM series."))


# ============================================================ 8. TECHNICAL (9)

_spec("moving_average", "technical", "SMA(CLOSE, 20)",
      lambda c: SMA(c.close, 20), scale_free=False,
      notes="A price level. Cross-sectionally this ranks stocks by nominal share "
            "price and essentially nothing else.")

_spec("exp_moving_average", "technical", "EMA(CLOSE, 20)",
      lambda c: EMA(c.close, 20), scale_free=False, notes="See moving_average.")

_spec("rsi", "technical", "RSI(14)", lambda c: RSI(c.close, 14))

_spec("macd", "technical", "(EMA(CLOSE, 12) - EMA(CLOSE, 26))",
      lambda c: ops.MACD(c.close), scale_free=False)

_spec("bollinger_bands", "technical", "UPPER_BAND - LOWER_BAND",
      lambda c: _boll_span(c), scale_free=False,
      notes="Raw band span in dollars (= 4 * STD(CLOSE,20)); monotone in std_dev.")

_spec("stochastic_oscillator", "technical",
      "((CLOSE - MIN(LOW,14)) / (MAX(HIGH,14) - MIN(LOW,14))) * 100",
      lambda c: ops.STOCHASTIC(c.high, c.low, c.close, 14))

_spec("atr_technical", "technical", "ATR(14)",
      lambda c: ATR(c.high, c.low, c.close, 14), scale_free=False,
      duplicate_of="atr", notes="A.3 lists ATR(14) under both Volatility and Technical.")

_spec("cci", "technical",
      "(TYPICAL_PRICE - SMA(TYPICAL_PRICE, 20)) / (0.015 * MEAN_DEV(TYPICAL_PRICE, 20))",
      lambda c: CCI(c.high, c.low, c.close, 20))

_spec("williams_r", "technical",
      "((MAX(HIGH,14) - CLOSE) / (MAX(HIGH,14) - MIN(LOW,14))) * -100",
      lambda c: WILLIAMS_R(c.high, c.low, c.close, 14))


def _boll_span(c: AlphaContext):
    upper, lower = BOLLINGER(c.close, 20, 2.0)
    return upper - lower


# ======================================================== 9. MACRO ECONOMICS (10)
# Every factor here is identical across stocks on a given day. They are computed
# so the count matches A.3 and so the degeneracy is demonstrated rather than
# asserted -- see the module docstring in data/macro.py.

_MACRO_LAG = 21  # A.3's unspecified "n"; one month of trading days.


def _macro_diff(series: str):
    def fn(c: AlphaContext):
        x = c.macro(series)
        return x - DELAY(x, _MACRO_LAG)
    return fn


_spec("interest_rate", "macro", "INTEREST_RATE - DELAY(INTEREST_RATE, n)",
      _macro_diff("interest_rate_10y"),
      notes="10-year Treasury yield (^TNX), n=21 trading days. Cross-sectionally "
            "constant: contributes nothing to a stock ranking.")

_spec("short_rate", "macro", "INTEREST_RATE - DELAY(INTEREST_RATE, n)",
      _macro_diff("interest_rate_3m"),
      notes="13-week T-bill (^IRX). Stands in for A.3's separate macro entries that "
            "require FRED. Cross-sectionally constant.")

_spec("implied_vol_change", "macro", "CCI - DELAY(CCI, n)  [proxy]",
      _macro_diff("implied_volatility"),
      notes="VIX change, used as the reachable stand-in for A.3's Consumer "
            "Confidence Index. Cross-sectionally constant.")

_spec("dollar_index_change", "macro", "FX_RESERVES - DELAY(FX_RESERVES, n)  [proxy]",
      _macro_diff("dollar_index"),
      notes="Dollar index (DXY) change, proxying A.3's FX reserves. Cross-sectionally constant.")

_spec("oil_price_change", "macro", "IPI - DELAY(IPI, n)  [proxy]",
      _macro_diff("oil_price"),
      notes="Crude oil, proxying industrial production. Cross-sectionally constant.")

_spec("gold_price_change", "macro", "CPI - DELAY(CPI, n)  [proxy]",
      _macro_diff("gold_price"),
      notes="Gold, proxying inflation. Cross-sectionally constant.")

for _mname, _mformula, _mreason in [
    ("gdp_growth", "GDP - DELAY(GDP, n)", "FRED GDPC1 unreachable from this environment"),
    ("unemployment_rate", "UNEMPLOYMENT_RATE - DELAY(UNEMPLOYMENT_RATE, n)", "FRED UNRATE unreachable"),
    ("retail_sales_growth", "RETAIL_SALES - DELAY(RETAIL_SALES, n)", "FRED RSAFS unreachable"),
    ("housing_starts_growth", "HOUSING_STARTS - DELAY(HOUSING_STARTS, n)", "FRED HOUST unreachable"),
]:
    _spec(_mname, "macro", _mformula, lambda c: c.panel.empty(), available=False,
          notes=f"{_mreason}. Would in any case be cross-sectionally constant.")
