"""Appendix A.8: top-k / drop-n daily portfolio construction.

The rule: rank every tradeable stock by the composite alpha, hold the top ``k``
equally weighted, and change at most ``n`` names a day. The turnover cap is what
makes the strategy implementable -- an unconstrained daily top-13 out of ~450
names would turn over most of the book most days.

Implementation follows the standard drop-n reading: each day, sell the ``n``
worst-ranked names currently held that have fallen out of the top ``k``, and buy
the best-ranked names not held to refill. Both legs are capped at ``n``, so
daily one-way turnover cannot exceed ``n / k``.

Two things the paper does not model, both first-order here:

**Execution lag.** A signal computed from the close of ``t`` cannot be traded at
the close of ``t``. With ``execution_lag_days = 1`` the book decided at ``t`` is
executed at the close of ``t+1`` and earns the return of ``t+2`` onward. Setting
it to 0 reproduces the optimistic same-bar assumption so the cost of that
assumption can be measured rather than argued about.

**Transaction costs.** At the paper's own stated ~38% daily turnover, 10bps
round-trip is roughly 19% a year of drag against a claimed 53% return. Costs are
charged on traded notional and every result is reported gross and net.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestResult:
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    """One-way turnover: 0.5 * sum |weight change|."""

    costs: pd.Series
    weights: pd.DataFrame
    holdings_count: pd.Series
    n_trades: pd.Series

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.gross_returns.index


def build_topk_dropn_weights(
    scores: pd.DataFrame, tradeable: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.Series]:
    """Target weights from a composite alpha panel, honouring the drop-n cap.

    Returns ``(weights, trade_counts)``. The trade count is returned rather than
    stashed on ``weights.attrs``: pandas propagates ``attrs`` through arithmetic,
    and ``pd.concat`` then compares them with ``==``, which raises on a Series
    ("truth value is ambiguous"). Carrying a Series in ``attrs`` therefore poisons
    every downstream concat of the resulting returns.

    ``scores`` is (date x ticker); higher is better. Rows where a name is not
    tradeable are ignored, and a held name that becomes untradeable is dropped
    immediately regardless of the turnover cap -- a suspended or delisted stock
    is not a position you get to keep.
    """
    k, n = cfg.portfolio.top_k, cfg.portfolio.drop_n
    dates = scores.index
    tickers = scores.columns
    weights = pd.DataFrame(0.0, index=dates, columns=tickers)
    trade_counts = pd.Series(0, index=dates, dtype=int)

    held: list[str] = []
    for date in dates:
        row = scores.loc[date].where(tradeable.loc[date])
        ranked = row.dropna().sort_values(ascending=False)
        if ranked.empty:
            # Nothing rankable: carry the book but drop anything untradeable.
            held = [t for t in held if bool(tradeable.loc[date, t])]
            if held:
                weights.loc[date, held] = 1.0 / len(held)
            continue

        ranked_names = list(ranked.index)

        if not held:
            # Building the book from cash. The drop-n cap governs how fast an
            # *existing* portfolio may be rotated, not how long it takes to
            # invest in the first place -- ramping in n names a day would leave
            # the book underweight for k/n sessions and inflate turnover as the
            # equal weights kept shrinking.
            held = ranked_names[:k]
            weights.loc[date, held] = 1.0 / len(held)
            trade_counts.loc[date] = len(held)
            continue

        # Forced exits first: no longer tradeable, or no longer scored. These
        # are not discretionary and are not subject to the turnover cap.
        survivors = [t for t in held if t in ranked.index]
        forced_out = len(held) - len(survivors)

        # Discretionary exits: the n worst-ranked survivors that have fallen out
        # of the target top k.
        rank_of = {t: i for i, t in enumerate(ranked_names)}
        outside = [t for t in survivors if t not in ranked_names[:k]]
        outside.sort(key=lambda t: rank_of[t], reverse=True)  # worst first
        discretionary_out = outside[:n]

        keep = [t for t in survivors if t not in discretionary_out]
        # Refill to k from the best-ranked names not already held. By
        # construction room == forced_out + len(discretionary_out), so buys are
        # capped at n plus whatever was forced out.
        room = max(k - len(keep), 0)
        candidates = [t for t in ranked_names if t not in keep]
        buys = candidates[:room]

        held = keep + buys
        if held:
            weights.loc[date, held] = 1.0 / len(held)
        trade_counts.loc[date] = len(discretionary_out) + forced_out + len(buys)

    return weights, trade_counts


def run_portfolio(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cfg: Config,
    trade_counts: pd.Series | None = None,
) -> BacktestResult:
    """Turn target weights into a realised return series, net of costs.

    ``weights.loc[t]`` is the book decided using information through the close of
    ``t``. It is executed ``execution_lag_days`` sessions later and earns the
    following session's return, so the effective holding on day ``s`` is
    ``weights.loc[s - 1 - lag]``.
    """
    lag = cfg.portfolio.execution_lag_days
    effective = weights.shift(1 + lag).fillna(0.0)
    aligned_returns = returns.reindex_like(effective).fillna(0.0)

    gross = (effective * aligned_returns).sum(axis=1)

    # Weights drift with returns between rebalances; trading only has to cover
    # the gap between the drifted book and the new target, not the whole book.
    drifted = effective.shift(1).fillna(0.0) * (1.0 + aligned_returns.shift(1).fillna(0.0))
    drift_total = drifted.sum(axis=1)
    drifted = drifted.div(drift_total.where(drift_total > 0), axis=0).fillna(0.0)

    weight_change = (effective - drifted).abs().sum(axis=1)
    turnover = 0.5 * weight_change

    one_way_bps = cfg.portfolio.cost_bps + cfg.portfolio.slippage_bps
    costs = weight_change * one_way_bps / 1e4
    net = gross - costs

    if trade_counts is None:
        trade_counts = pd.Series(0, index=weights.index)
    return BacktestResult(
        gross_returns=gross,
        net_returns=net,
        turnover=turnover,
        costs=costs,
        weights=effective,
        holdings_count=(effective > 0).sum(axis=1),
        n_trades=trade_counts.reindex(gross.index).fillna(0).astype(int),
    )
