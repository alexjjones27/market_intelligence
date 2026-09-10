"""Risk Preference Agent (RPA).

The paper gives only ``rho_ij = f_risk(alpha, market state)`` (equation 6) and
never defines ``f_risk``. Since this is the component most easily
reverse-engineered into working, the definition is fixed up front, kept simple,
and documented here in full:

``f_risk`` blends three penalties, each measured on the long-short quantile book
that the alpha itself implies, over a trailing window that is fully observable
at the scoring date:

1. **Realised volatility** of the factor's long-short spread return.
2. **Maximum drawdown** of that spread's cumulative return over the window.
3. **IC sign stability** -- the share of ``ic_stability_window``-long sub-blocks
   whose mean IC carries the same sign as the window overall. A factor whose
   sign flips every quarter is not a factor, it is noise with a long memory.

Each component is converted to a *desirability* in [0, 1] by cross-sectional
percentile rank across alphas at that date, then combined with the configured
weights. Higher is always better, so the paper's ``wc*theta + wr*rho`` remains a
maximisation with no sign gymnastics.

Percentile-ranking both agents' outputs is what makes their weighted sum
meaningful at all: a raw mean IC lives around 0.02 while a raw drawdown lives
around 0.30, and adding them with weights 0.6/0.4 would otherwise be dominated
by whichever happens to carry the larger units. The cost of this choice is that
Algorithm 1's threshold X becomes a percentile rather than an absolute level,
which is noted where the threshold is applied.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.agents.factor_stats import (
    FactorStats,
    observable_ic_window,
    observable_spread_window,
)
from alpha_lab.config import Config


@dataclass(frozen=True)
class RiskScores:
    score: pd.Series
    volatility: pd.Series
    max_drawdown: pd.Series
    ic_stability: pd.Series


def _max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline of the cumulative return path."""
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    equity = (1.0 + clean).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def _ic_sign_stability(ic: pd.Series, block: int) -> float:
    """Share of sub-blocks whose mean IC agrees in sign with the whole window."""
    clean = ic.dropna()
    if len(clean) < 2 * block:
        return np.nan
    overall = np.sign(clean.mean())
    if overall == 0:
        return 0.0
    blocks = [clean.iloc[i : i + block] for i in range(0, len(clean) - block + 1, block)]
    signs = [np.sign(b.mean()) for b in blocks if len(b) == block]
    if not signs:
        return np.nan
    return float(np.mean([s == overall for s in signs]))


def _to_desirability(values: pd.Series, higher_is_better: bool) -> pd.Series:
    """Percentile-rank across alphas into [0, 1], 1 = most desirable."""
    ranked = values.rank(pct=True)
    return ranked if higher_is_better else 1.0 - ranked


def score_risk(stats: FactorStats, cfg: Config, as_of: pd.Timestamp) -> RiskScores:
    """Risk desirability for every alpha, as knowable on ``as_of``."""
    spread_window = observable_spread_window(stats.spread, as_of, cfg.rpa.lookback_days)
    ic_window = observable_ic_window(stats.ic, as_of, stats.horizon, cfg.rpa.lookback_days)

    trading_days = cfg.evaluation.trading_days
    n_obs = spread_window.notna().sum()

    volatility = spread_window.std() * np.sqrt(trading_days)
    drawdown = spread_window.apply(_max_drawdown)
    stability = ic_window.apply(lambda s: _ic_sign_stability(s, cfg.rpa.ic_stability_window))

    insufficient = n_obs < cfg.rpa.min_periods
    volatility = volatility.where(~insufficient)
    drawdown = drawdown.where(~insufficient)
    stability = stability.where(~insufficient)

    # Low volatility is good; a shallow (less negative) drawdown is good;
    # a stable IC sign is good.
    components = (
        cfg.rpa.w_volatility * _to_desirability(volatility, higher_is_better=False)
        + cfg.rpa.w_drawdown * _to_desirability(drawdown, higher_is_better=True)
        + cfg.rpa.w_ic_stability * _to_desirability(stability, higher_is_better=True)
    )
    total_weight = cfg.rpa.w_volatility + cfg.rpa.w_drawdown + cfg.rpa.w_ic_stability
    score = components / total_weight if total_weight > 0 else components

    return RiskScores(
        score=score.astype(float).where(~insufficient),
        volatility=volatility.astype(float),
        max_drawdown=drawdown.astype(float),
        ic_stability=stability.astype(float),
    )
