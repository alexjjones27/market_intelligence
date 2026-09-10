"""Synthetic fixtures: fast, deterministic, and independent of network access."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.config import Config
from alpha_lab.data.panel import MarketPanel


def make_prices(n_days: int = 400, n_tickers: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    # Deliberately spread starting prices over two orders of magnitude so
    # scale-dependence in the A.3 formulas is visible in tests.
    start = rng.uniform(5, 500, size=n_tickers)
    steps = rng.normal(0.0004, 0.018, size=(n_days, n_tickers))
    close = pd.DataFrame(start * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=tickers)

    intraday = np.abs(rng.normal(0.008, 0.004, size=(n_days, n_tickers)))
    high = close * (1 + intraday)
    low = close * (1 - intraday)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.004, size=(n_days, n_tickers)))
    open_ = open_.clip(lower=low, upper=high)
    volume = pd.DataFrame(
        rng.lognormal(14, 0.6, size=(n_days, n_tickers)), index=dates, columns=tickers
    )
    return dict(open=open_, high=high, low=low, close=close, volume=volume,
                adj_close=close, dates=dates, tickers=tickers)


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def panel() -> MarketPanel:
    p = make_prices()
    dates, tickers = p["dates"], p["tickers"]
    fields = {k: p[k] for k in ("open", "high", "low", "close", "adj_close", "volume")}
    fields["vwap"] = (fields["high"] + fields["low"] + fields["close"]) / 3
    fields["returns"] = fields["adj_close"].pct_change(fill_method=None)

    rng = np.random.default_rng(1)
    # Step-function fundamentals that only change on 'filing' dates, mimicking
    # the real panel's quarterly refresh.
    fundamental_scales = {
        "eps": 5.0, "revenue": 1e10, "total_equity": 5e9, "total_assets": 2e10,
        "book_value_per_share": 30.0, "revenue_per_share": 60.0,
        "cash_flow_per_share": 8.0, "dividends_per_share": 1.5,
        "shares_outstanding": 5e8, "net_income": 1e9, "operating_income": 1.2e9,
        "gross_profit": 4e9, "cogs": 6e9, "inventory": 1e9, "receivables": 2e9,
        "payables": 1.5e9, "total_debt": 3e9, "ebit": 1.2e9, "ebitda": 1.5e9,
        "interest_expense": 1e8, "operating_cash_flow": 2e9,
        "retained_earnings": 8e9, "book_value": 5e9, "market_cap": 1e11,
        "enterprise_value": 1.05e11,
    }
    for name, scale in fundamental_scales.items():
        base = rng.uniform(0.5, 1.5, size=len(tickers)) * scale
        vals = np.repeat(base[None, :], len(dates), axis=0)
        for q in range(1, 8):  # quarterly steps
            idx = q * 63
            if idx < len(dates):
                vals[idx:] *= rng.uniform(0.97, 1.06, size=len(tickers))
        fields[name] = pd.DataFrame(vals, index=dates, columns=tickers)

    tradeable = pd.DataFrame(True, index=dates, columns=tickers)
    return MarketPanel(
        fields=fields,
        macro=pd.DataFrame(
            {"interest_rate_10y": np.linspace(2, 4, len(dates)),
             "interest_rate_3m": np.linspace(1, 5, len(dates)),
             "implied_volatility": np.linspace(15, 25, len(dates)),
             "dollar_index": np.linspace(90, 105, len(dates)),
             "oil_price": np.linspace(50, 90, len(dates)),
             "gold_price": np.linspace(1500, 2000, len(dates))},
            index=dates),
        benchmarks=pd.DataFrame(index=dates),
        sectors=pd.Series(["Tech"] * len(tickers), index=tickers),
        membership=tradeable.copy(),
        liquid=tradeable.copy(),
        tradeable=tradeable,
        returns=fields["returns"],
    )
