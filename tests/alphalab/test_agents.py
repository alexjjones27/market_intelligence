"""Stage 2 observability and selection mechanics.

The single most dangerous line in Stage 2 is the one that decides which IC
observations an agent is allowed to see. An IC computed at date t uses the
return from t to t+N, so it is not knowable until t+N; using it to score at t
lets the confidence agent read returns that have not happened. These tests pin
that boundary, and then check that the whole selection step is invariant to
truncating the panel at the scoring date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.factor_stats import (
    FactorStats, daily_rank_ic, forward_return, observable_ic_window,
    observable_spread_window, quantile_spread_return,
)
from alpha_lab.agents.selection import select_alphas
from alpha_lab.alphas.library import registry
from alpha_lab.alphas.registry import compute_alphas
from alpha_lab.agents.factor_stats import compute_factor_stats


def test_observable_ic_window_excludes_unfinished_forward_returns():
    dates = pd.bdate_range("2021-01-04", periods=100)
    ic = pd.DataFrame({"a": np.arange(100.0)}, index=dates)
    as_of = dates[80]
    window = observable_ic_window(ic, as_of, horizon=5, lookback=None)
    # The newest usable row must be 5 sessions back: its forward return ended
    # on as_of itself. Row 80 would need returns through session 85.
    assert window.index.max() == dates[75]
    assert as_of not in window.index


def test_observable_ic_window_respects_lookback():
    dates = pd.bdate_range("2021-01-04", periods=300)
    ic = pd.DataFrame({"a": np.arange(300.0)}, index=dates)
    window = observable_ic_window(ic, dates[250], horizon=5, lookback=60)
    assert len(window) == 60
    assert window.index.max() == dates[245]


def test_observable_ic_window_is_empty_before_the_first_horizon():
    dates = pd.bdate_range("2021-01-04", periods=10)
    ic = pd.DataFrame({"a": np.arange(10.0)}, index=dates)
    assert observable_ic_window(ic, dates[3], horizon=5, lookback=None).empty


def test_observable_spread_window_includes_today():
    """Spread returns are realised, so today's is knowable at today's close."""
    dates = pd.bdate_range("2021-01-04", periods=50)
    spread = pd.DataFrame({"a": np.arange(50.0)}, index=dates)
    window = observable_spread_window(spread, dates[30], lookback=None)
    assert window.index.max() == dates[30]


def test_forward_return_is_not_knowable_today():
    dates = pd.bdate_range("2021-01-04", periods=10)
    adj = pd.DataFrame({"A": np.linspace(100, 110, 10)}, index=dates)

    class Stub:
        def __getitem__(self, key):
            return adj

    fwd = forward_return(Stub(), 5)
    expected = adj["A"].iloc[5] / adj["A"].iloc[0] - 1
    assert fwd["A"].iloc[0] == pytest.approx(expected)
    # The last 5 rows cannot be computed at all.
    assert fwd["A"].iloc[-5:].isna().all()


def test_daily_rank_ic_detects_a_perfect_signal():
    dates = pd.bdate_range("2021-01-04", periods=3)
    cols = [f"T{i}" for i in range(30)]
    alpha = pd.DataFrame(np.tile(np.arange(30.0), (3, 1)), index=dates, columns=cols)
    fwd = alpha * 2.0  # perfectly rank-correlated
    ic = daily_rank_ic(alpha, fwd, min_obs=20)
    assert ic.dropna().round(6).eq(1.0).all()
    ic_inverted = daily_rank_ic(alpha, -fwd, min_obs=20)
    assert ic_inverted.dropna().round(6).eq(-1.0).all()


def test_daily_rank_ic_requires_a_minimum_cross_section():
    dates = pd.bdate_range("2021-01-04", periods=2)
    alpha = pd.DataFrame(np.tile(np.arange(5.0), (2, 1)), index=dates)
    assert daily_rank_ic(alpha, alpha, min_obs=20).isna().all()


def test_quantile_spread_is_indexed_by_the_day_the_return_happened():
    """spread[t] must be earned by a book formed at t-1, so it is known at t."""
    dates = pd.bdate_range("2021-01-04", periods=4)
    cols = [f"T{i}" for i in range(40)]
    alpha = pd.DataFrame(np.tile(np.arange(40.0), (4, 1)), index=dates, columns=cols)

    # Derive the move from the same quantile masks the function uses, so this
    # asserts the *timing* rather than a particular tie-breaking convention at
    # the quintile boundary.
    ranks = alpha.rank(axis=1, pct=True)
    returns = pd.DataFrame(0.0, index=dates, columns=cols)
    returns.iloc[2] = np.where(ranks.iloc[2] >= 0.8, 0.10,
                               np.where(ranks.iloc[2] <= 0.2, -0.10, 0.0))

    spread = quantile_spread_return(alpha, returns, quantile=0.2, min_obs=20)
    # The move happens on day 2, so the spread must be booked on day 2 -- not on
    # day 1 (when the book was formed) and not on day 3.
    assert spread.iloc[2] == pytest.approx(0.20)
    assert spread.iloc[1] == pytest.approx(0.0)
    assert spread.iloc[3] == pytest.approx(0.0)


def test_selection_is_unchanged_by_data_after_the_scoring_date(panel, cfg):
    """Stage 2 truncation invariance: the selection made on a date must not
    depend on anything that happens after it."""
    import dataclasses
    from tests.alphalab.test_leakage import truncate

    alphas = compute_alphas(registry, panel, cfg)
    as_of = panel.dates[320]

    stats_full = compute_factor_stats(alphas, panel, cfg)
    cut_panel = truncate(panel, as_of)
    alphas_cut = compute_alphas(registry, cut_panel, cfg)
    stats_cut = compute_factor_stats(alphas_cut, cut_panel, cfg)

    shared = sorted(set(stats_full.alphas) & set(stats_cut.alphas))
    full = select_alphas(
        dataclasses.replace(stats_full, ic=stats_full.ic[shared], spread=stats_full.spread[shared]),
        registry, cfg, as_of,
    )
    cut = select_alphas(
        dataclasses.replace(stats_cut, ic=stats_cut.ic[shared], spread=stats_cut.spread[shared]),
        registry, cfg, as_of,
    )
    assert full.alphas == cut.alphas


def test_selection_takes_at_most_one_alpha_per_category(panel, cfg):
    alphas = compute_alphas(registry, panel, cfg)
    stats = compute_factor_stats(alphas, panel, cfg)
    selection = select_alphas(stats, registry, cfg, panel.dates[350])
    categories = [registry[a].category for a in selection.alphas]
    assert len(categories) == len(set(categories))


def test_disabling_both_agents_is_rejected(panel, cfg):
    alphas = compute_alphas(registry, panel, cfg)
    stats = compute_factor_stats(alphas, panel, cfg)
    bad = cfg.replace(selection={"use_csa": False, "use_rpa": False})
    with pytest.raises(ValueError):
        select_alphas(stats, registry, bad, panel.dates[350])
