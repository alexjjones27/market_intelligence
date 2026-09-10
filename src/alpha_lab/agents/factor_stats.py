"""Per-alpha time series that Stage 2's two agents both read from.

Two panels are computed once over the whole history and then sliced by the
walk-forward folds:

* ``ic``     -- the daily cross-sectional rank IC of each alpha against the
                forward N-day return.
* ``spread`` -- the daily return of a long-short quantile portfolio formed on
                each alpha, which is what the risk agent measures risk on.

**Observability is the whole game here.** These two panels become knowable at
different times and must never be mixed up:

``ic[t]`` correlates the alpha observed at ``t`` with the return from ``t`` to
``t + N``. Nobody can compute it until ``t + N``. So an agent scoring at date
``d`` may only use ``ic`` rows up to ``d - N``, and :func:`observable_ic_window`
exists so no caller has to re-derive that offset by hand.

``spread[t]`` is the return *realised on* day ``t`` by a portfolio formed from
the alpha at ``t - 1``. It is known at the close of ``t``, so an agent scoring
at ``d`` may use rows through ``d`` itself.

Getting this wrong is subtle and expensive: using ``ic`` rows up to ``d``
instead of ``d - N`` lets the confidence score peek at returns that have not
happened yet, which is precisely the kind of leak that manufactures a
spectacular backtest.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.alphas.registry import AlphaPanel
from alpha_lab.config import Config
from alpha_lab.data.panel import MarketPanel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FactorStats:
    ic: pd.DataFrame
    """index = date t, columns = alpha. Knowable only at t + horizon."""

    spread: pd.DataFrame
    """index = date t, columns = alpha. Knowable at the close of t."""

    horizon: int
    coverage: pd.DataFrame

    @property
    def alphas(self) -> list[str]:
        return list(self.ic.columns)


def forward_return(panel: MarketPanel, horizon: int) -> pd.DataFrame:
    """Return from the close of ``t`` to the close of ``t + horizon``.

    Built from the split- and dividend-adjusted close so a split is not booked
    as a crash and dividend payers are not systematically penalised. The value
    at ``t`` is *not* observable at ``t``; every consumer must lag it.
    """
    adj = panel["adj_close"]
    return adj.shift(-horizon) / adj - 1.0


def daily_rank_ic(alpha: pd.DataFrame, fwd: pd.DataFrame, min_obs: int = 20) -> pd.Series:
    """Per-day Spearman correlation between an alpha and the forward return.

    Vectorised over dates: ranking row-wise and taking a row-wise Pearson
    correlation of the ranks is exactly Spearman, and avoids 1,600 scipy calls
    per alpha. Days with fewer than ``min_obs`` paired observations return NaN
    rather than a correlation computed off a handful of names.
    """
    ra = alpha.rank(axis=1)
    rb = fwd.rank(axis=1)
    both = ra.notna() & rb.notna()
    ra, rb = ra.where(both), rb.where(both)
    n = both.sum(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    denom = np.sqrt((ra**2).sum(axis=1) * (rb**2).sum(axis=1))
    ic = (ra * rb).sum(axis=1) / denom.where(denom > 0)
    return ic.where(n >= min_obs)


def quantile_spread_return(
    alpha: pd.DataFrame,
    returns: pd.DataFrame,
    quantile: float = 0.2,
    min_obs: int = 20,
) -> pd.Series:
    """Daily return of an equal-weight long-top / short-bottom quantile book.

    The alpha observed at ``t`` sets the book; the return realised on ``t + 1``
    is what it earns. The output is indexed by the date the return *occurred*,
    so ``spread[t]`` is known at the close of ``t``.
    """
    ranks = alpha.rank(axis=1, pct=True)
    counts = alpha.notna().sum(axis=1)
    valid = counts >= min_obs

    long_mask = (ranks >= 1.0 - quantile) & valid.to_numpy()[:, None]
    short_mask = (ranks <= quantile) & valid.to_numpy()[:, None]

    # Weights formed at t are held through t+1's return.
    next_returns = returns.shift(-1)
    long_leg = next_returns.where(long_mask).mean(axis=1)
    short_leg = next_returns.where(short_mask).mean(axis=1)
    spread = (long_leg - short_leg).where(valid)
    # Re-index onto the date the return actually happened.
    return spread.shift(1)


def observable_ic_window(
    ic: pd.DataFrame, as_of: pd.Timestamp, horizon: int, lookback: int | None
) -> pd.DataFrame:
    """IC rows an agent scoring on ``as_of`` is actually allowed to see.

    The newest usable row is ``as_of - horizon`` trading days: its forward
    return had to have completed on or before ``as_of``. ``lookback=None``
    gives an expanding window.
    """
    usable = ic.index[ic.index <= as_of]
    if len(usable) <= horizon:
        return ic.iloc[:0]
    cutoff = usable[-(horizon + 1)]
    window = ic.loc[:cutoff]
    if lookback is not None and len(window) > lookback:
        window = window.iloc[-lookback:]
    return window


def observable_spread_window(
    spread: pd.DataFrame, as_of: pd.Timestamp, lookback: int | None
) -> pd.DataFrame:
    """Spread rows observable at ``as_of`` (realised returns, so through today)."""
    window = spread.loc[spread.index <= as_of]
    if lookback is not None and len(window) > lookback:
        window = window.iloc[-lookback:]
    return window


def compute_factor_stats(
    alpha_panel: AlphaPanel,
    panel: MarketPanel,
    cfg: Config,
    names: list[str] | None = None,
) -> FactorStats:
    """Compute the IC and spread panels for every usable alpha, once."""
    names = names or alpha_panel.usable()
    horizon = cfg.csa.ic_horizon
    fwd = forward_return(panel, horizon).where(panel.tradeable)
    daily_returns = panel["returns"].where(panel.tradeable)

    ic_cols: dict[str, pd.Series] = {}
    spread_cols: dict[str, pd.Series] = {}
    for name in names:
        alpha = alpha_panel.normalized[name]
        ic_cols[name] = daily_rank_ic(alpha, fwd, cfg.alphas.min_cross_section)
        spread_cols[name] = quantile_spread_return(
            alpha, daily_returns, cfg.rpa.spread_quantile, cfg.alphas.min_cross_section
        )

    ic = pd.DataFrame(ic_cols, index=panel.dates)
    spread = pd.DataFrame(spread_cols, index=panel.dates)

    coverage = pd.DataFrame(
        {
            "alpha": names,
            "category": [alpha_panel.registry[n].category for n in names],
            "n_ic_days": [int(ic[n].notna().sum()) for n in names],
            "mean_ic": [float(ic[n].mean()) for n in names],
            "abs_mean_ic": [abs(float(ic[n].mean())) for n in names],
            "ic_std": [float(ic[n].std()) for n in names],
            "ic_ir": [
                float(ic[n].mean() / ic[n].std()) if ic[n].std() > 0 else np.nan
                for n in names
            ],
            "spread_ann_return": [float(spread[n].mean() * cfg.evaluation.trading_days) for n in names],
            "spread_ann_vol": [
                float(spread[n].std() * np.sqrt(cfg.evaluation.trading_days)) for n in names
            ],
        }
    ).sort_values("abs_mean_ic", ascending=False)

    logger.info("factor stats: %d alphas, horizon %dd", len(names), horizon)
    return FactorStats(ic=ic, spread=spread, horizon=horizon, coverage=coverage)
