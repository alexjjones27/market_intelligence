"""Performance metrics.

All ratios are reported **annualised**. That is worth stating because the
paper's own numbers are not internally consistent on this point: Table 4 shows
Sharpe 0.287 alongside a 53.17% return and 0.762% volatility, which only
reconciles if that Sharpe is a *daily* figure (0.287 x sqrt(252) is about 4.6
annualised), while Table 7 reports 1.94 and Tables 9-10 report 11.39 and 13.33
for what is described as the same model. Everything here is annualised with an
explicit trading-day count so there is no ambiguity about which convention is in
force.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(returns: pd.Series) -> pd.Series:
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def cumulative_return(returns: pd.Series) -> float:
    r = _clean(returns)
    return float((1.0 + r).prod() - 1.0) if len(r) else np.nan


def annualised_return(returns: pd.Series, trading_days: int = 252) -> float:
    """Geometric (CAGR) annualisation, not the arithmetic mean x 252.

    On a volatile daily series the arithmetic version overstates badly: it is
    how a 59% half-year becomes a reported '192.27% annual return' in the
    paper's Table 6, where compounding gives 153%.
    """
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    total = (1.0 + r).prod()
    if total <= 0:
        return -1.0
    return float(total ** (trading_days / len(r)) - 1.0)


def annualised_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    r = _clean(returns)
    return float(r.std() * np.sqrt(trading_days)) if len(r) > 1 else np.nan


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, trading_days: int = 252) -> float:
    r = _clean(returns)
    if len(r) < 2 or r.std() == 0:
        return np.nan
    excess = r - risk_free / trading_days
    return float(excess.mean() / r.std() * np.sqrt(trading_days))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, trading_days: int = 252) -> float:
    """Excess return per unit of *downside* deviation."""
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    excess = r - risk_free / trading_days
    downside = excess[excess < 0]
    if len(downside) < 2 or downside.std() == 0:
        return np.nan
    return float(excess.mean() / downside.std() * np.sqrt(trading_days))


def max_drawdown(returns: pd.Series) -> float:
    r = _clean(returns)
    if r.empty:
        return np.nan
    equity = (1.0 + r).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def calmar_ratio(returns: pd.Series, trading_days: int = 252) -> float:
    dd = max_drawdown(returns)
    if dd is np.nan or dd == 0:
        return np.nan
    return float(annualised_return(returns, trading_days) / abs(dd))


def hit_rate(returns: pd.Series) -> float:
    r = _clean(returns)
    return float((r > 0).mean()) if len(r) else np.nan


def summarise(
    returns: pd.Series,
    *,
    label: str = "",
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
    risk_free: float = 0.0,
    trading_days: int = 252,
) -> dict:
    r = _clean(returns)
    out = {
        "label": label,
        "n_days": int(len(r)),
        "cum_return": cumulative_return(r),
        "ann_return": annualised_return(r, trading_days),
        "ann_volatility": annualised_volatility(r, trading_days),
        "sharpe": sharpe_ratio(r, risk_free, trading_days),
        "sortino": sortino_ratio(r, risk_free, trading_days),
        "calmar": calmar_ratio(r, trading_days),
        "max_drawdown": max_drawdown(r),
        "hit_rate": hit_rate(r),
    }
    if turnover is not None:
        t = _clean(turnover)
        out["mean_daily_turnover"] = float(t.mean()) if len(t) else np.nan
        out["ann_turnover"] = float(t.mean() * trading_days) if len(t) else np.nan
    if costs is not None:
        c = _clean(costs)
        out["ann_cost_drag"] = float(c.mean() * trading_days) if len(c) else np.nan
    return out


def summarise_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
