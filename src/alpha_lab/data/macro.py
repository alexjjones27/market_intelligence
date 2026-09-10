"""Macroeconomic series for the Appendix A.3 "Macro Economics" category.

Two things need saying up front, because they change how these factors should
be read:

1. **FRED is unreachable from this environment** (``fred.stlouisfed.org`` times
   out through the outbound proxy), so GDP, CPI, unemployment, industrial
   production, retail sales, housing starts, consumer confidence, trade balance
   and FX reserves -- nine of the ten A.3 macro alphas -- cannot be sourced as
   specified. What is available are market-based proxies from Yahoo: the 10-year
   and 13-week Treasury yields, VIX, the dollar index, oil and gold. Only
   "Interest Rate" maps onto A.3 directly; the rest are flagged unavailable.

2. **Every A.3 macro alpha is cross-sectionally constant.** ``GDP - DELAY(GDP, n)``
   takes the same value for every stock on a given day. In a strategy that
   *ranks stocks against each other* -- which is exactly what Appendix A.8's
   top-k/drop-n construction does -- a constant column changes no ranking, and
   its cross-sectional IC is not merely zero but undefined (the Spearman
   correlation of a constant vector against anything is NaN).

   So the paper's ninth category cannot contribute to the paper's own portfolio
   rule, no matter what the market does. This is a property of the design, not
   of our data access, and it would hold identically with a full FRED feed. The
   pipeline computes these factors anyway, detects the degeneracy explicitly
   (see ``alphas.registry.degenerate_alphas``) and reports it rather than
   letting a NaN IC quietly drop the category.

A macro series *can* be made cross-sectional by interacting it with per-stock
sensitivity (a rolling beta to the series). That is a real technique, but it is
not what A.3 specifies, so it is not silently substituted here.
"""
from __future__ import annotations

import logging

import pandas as pd

from alpha_lab.config import Config
from alpha_lab.data.prices import load_prices

logger = logging.getLogger(__name__)

# A.3 name -> Yahoo proxy. Everything else in A.3's macro list needs FRED.
MACRO_PROXIES: dict[str, str] = {
    "interest_rate_10y": "^TNX",
    "interest_rate_3m": "^IRX",
    "implied_volatility": "^VIX",
    "dollar_index": "DX-Y.NYB",
    "oil_price": "CL=F",
    "gold_price": "GC=F",
}

UNAVAILABLE_MACRO: dict[str, str] = {
    "gdp_growth": "needs FRED GDPC1 (unreachable from this environment)",
    "inflation_rate": "needs FRED CPIAUCSL",
    "unemployment_rate": "needs FRED UNRATE",
    "industrial_production": "needs FRED INDPRO",
    "retail_sales": "needs FRED RSAFS",
    "housing_starts": "needs FRED HOUST",
    "consumer_confidence": "needs FRED UMCSENT",
    "trade_balance": "needs FRED BOPGSTB",
    "fx_reserves": "needs FRED TRESEGUSM052N",
}

BENCHMARKS: dict[str, str] = {
    # SPY rather than ^GSPC for the cap-weight line: ^GSPC is a price index and
    # excludes dividends, which understates it by roughly 1.8% a year and would
    # flatter any strategy compared against it. ^GSPC is kept for reference.
    "sp500_cap_weight": "SPY",
    "sp500_equal_weight": "RSP",
    "sp500_price_index": "^GSPC",
}


def load_macro(cfg: Config, trading_days: pd.DatetimeIndex, refresh: bool = False) -> pd.DataFrame:
    """Daily macro proxies aligned to the trading calendar.

    Forward-filled across gaps (a Treasury-market holiday is not new
    information), which is safe here because these are *observed* market levels
    with no restatement and no publication lag.
    """
    if cfg.data.macro_source == "none":
        return pd.DataFrame(index=trading_days)
    tickers = list(MACRO_PROXIES.values())
    try:
        prices = load_prices(cfg, tickers, refresh=refresh)
    except Exception as exc:
        logger.warning("macro proxy fetch failed (%s); macro category will be empty", exc)
        return pd.DataFrame(index=trading_days)

    close = prices["close"]
    out = pd.DataFrame(index=trading_days)
    for name, ticker in MACRO_PROXIES.items():
        if ticker in close.columns:
            out[name] = close[ticker].reindex(trading_days).ffill()
    logger.info("macro proxies loaded: %s", list(out.columns))
    return out


def load_benchmarks(cfg: Config, trading_days: pd.DatetimeIndex, refresh: bool = False) -> pd.DataFrame:
    """Benchmark total-return series (cap-weight and equal-weight S&P 500)."""
    try:
        prices = load_prices(cfg, list(BENCHMARKS.values()), refresh=refresh)
    except Exception as exc:
        logger.warning("benchmark fetch failed: %s", exc)
        return pd.DataFrame(index=trading_days)
    adj = prices["adj_close"]
    out = pd.DataFrame(index=trading_days)
    for name, ticker in BENCHMARKS.items():
        if ticker in adj.columns:
            out[name] = adj[ticker].reindex(trading_days).ffill()
    return out
