"""Top-k/drop-n construction, execution lag, and transaction costs.

These are the places where an optimistic backtest is usually manufactured: a
book that rebalances faster than the turnover cap allows, weights that earn the
same day's return they were computed from, or costs that quietly go missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.config import Config
from alpha_lab.portfolio.construction import build_topk_dropn_weights, run_portfolio


@pytest.fixture
def scores_frame():
    dates = pd.bdate_range("2021-01-04", periods=30)
    cols = [f"T{i:02d}" for i in range(40)]
    rng = np.random.default_rng(4)
    return pd.DataFrame(rng.normal(size=(30, 40)), index=dates, columns=cols)


@pytest.fixture
def tradeable_frame(scores_frame):
    return pd.DataFrame(True, index=scores_frame.index, columns=scores_frame.columns)


def test_holds_exactly_top_k_once_warmed_up(scores_frame, tradeable_frame):
    cfg = Config().replace(portfolio={"top_k": 13, "drop_n": 5})
    weights = build_topk_dropn_weights(scores_frame, tradeable_frame, cfg)
    counts = (weights > 0).sum(axis=1)
    assert counts.iloc[0] == 13          # first day fills straight to k
    assert (counts.iloc[1:] == 13).all()


def test_weights_are_equal_and_sum_to_one(scores_frame, tradeable_frame):
    cfg = Config()
    weights = build_topk_dropn_weights(scores_frame, tradeable_frame, cfg)
    row = weights.iloc[5]
    held = row[row > 0]
    assert row.sum() == pytest.approx(1.0)
    assert held.nunique() == 1


def test_daily_name_changes_respect_the_drop_n_cap(scores_frame, tradeable_frame):
    """Scores are reshuffled every day, so an uncapped rule would replace the
    whole book; the cap must hold it to n."""
    cfg = Config().replace(portfolio={"top_k": 13, "drop_n": 5})
    weights = build_topk_dropn_weights(scores_frame, tradeable_frame, cfg)
    held = weights > 0
    for i in range(2, len(weights)):
        previous = set(held.columns[held.iloc[i - 1]])
        current = set(held.columns[held.iloc[i]])
        assert len(current - previous) <= cfg.portfolio.drop_n


def test_untradeable_names_are_dropped_even_past_the_cap(scores_frame, tradeable_frame):
    """A suspended or delisted name is not a position you get to keep."""
    cfg = Config().replace(portfolio={"top_k": 13, "drop_n": 1})
    tradeable = tradeable_frame.copy()
    weights_before = build_topk_dropn_weights(scores_frame, tradeable, cfg)
    held_day5 = list(weights_before.columns[weights_before.iloc[5] > 0])

    tradeable.iloc[6:, tradeable.columns.get_indexer(held_day5[:6])] = False
    weights = build_topk_dropn_weights(scores_frame, tradeable, cfg)
    still_held = set(weights.columns[weights.iloc[6] > 0]) & set(held_day5[:6])
    assert not still_held


def test_execution_lag_means_a_signal_cannot_earn_its_own_day(scores_frame, tradeable_frame):
    """With lag=1, the book decided at t is executed at t+1 and earns t+2."""
    dates = scores_frame.index
    cols = scores_frame.columns
    weights = pd.DataFrame(0.0, index=dates, columns=cols)
    weights.iloc[10, 0] = 1.0  # decided on day 10

    returns = pd.DataFrame(0.0, index=dates, columns=cols)
    returns.iloc[11, 0] = 0.50   # day 11: execution day, must NOT be earned
    returns.iloc[12, 0] = 0.20   # day 12: first day the position is on

    cfg = Config().replace(portfolio={"execution_lag_days": 1, "cost_bps": 0.0,
                                      "slippage_bps": 0.0})
    result = run_portfolio(weights, returns, cfg)
    assert result.gross_returns.iloc[11] == pytest.approx(0.0)
    assert result.gross_returns.iloc[12] == pytest.approx(0.20)


def test_zero_lag_still_never_earns_the_signal_day(scores_frame, tradeable_frame):
    """Even the optimistic setting trades at the close of t and earns t+1."""
    dates, cols = scores_frame.index, scores_frame.columns
    weights = pd.DataFrame(0.0, index=dates, columns=cols)
    weights.iloc[10, 0] = 1.0
    returns = pd.DataFrame(0.0, index=dates, columns=cols)
    returns.iloc[10, 0] = 0.50
    returns.iloc[11, 0] = 0.20

    cfg = Config().replace(portfolio={"execution_lag_days": 0, "cost_bps": 0.0,
                                      "slippage_bps": 0.0})
    result = run_portfolio(weights, returns, cfg)
    assert result.gross_returns.iloc[10] == pytest.approx(0.0)
    assert result.gross_returns.iloc[11] == pytest.approx(0.20)


def test_costs_are_charged_on_traded_notional():
    """Entering a full position from cash trades 100% of the book once."""
    dates = pd.bdate_range("2021-01-04", periods=6)
    cols = ["A", "B"]
    weights = pd.DataFrame(0.0, index=dates, columns=cols)
    weights.iloc[:, 0] = 1.0
    returns = pd.DataFrame(0.0, index=dates, columns=cols)

    cfg = Config().replace(portfolio={"execution_lag_days": 0, "cost_bps": 6.0,
                                      "slippage_bps": 4.0})
    result = run_portfolio(weights, returns, cfg)
    entry = result.costs[result.costs > 0]
    assert entry.iloc[0] == pytest.approx(10.0 / 1e4)   # 10bps on 100% traded
    # Once the book stops changing, costs stop.
    assert result.costs.iloc[-1] == pytest.approx(0.0)


def test_a_static_book_incurs_no_turnover_or_cost():
    dates = pd.bdate_range("2021-01-04", periods=10)
    cols = ["A", "B"]
    weights = pd.DataFrame(0.5, index=dates, columns=cols)
    returns = pd.DataFrame(0.0, index=dates, columns=cols)
    cfg = Config()
    result = run_portfolio(weights, returns, cfg)
    assert result.turnover.iloc[4:].abs().max() == pytest.approx(0.0)
    assert result.costs.iloc[4:].abs().max() == pytest.approx(0.0)


def test_net_return_is_gross_minus_cost(scores_frame, tradeable_frame):
    cfg = Config()
    weights = build_topk_dropn_weights(scores_frame, tradeable_frame, cfg)
    rng = np.random.default_rng(9)
    returns = pd.DataFrame(
        rng.normal(0, 0.01, size=scores_frame.shape),
        index=scores_frame.index, columns=scores_frame.columns,
    )
    result = run_portfolio(weights, returns, cfg)
    pd.testing.assert_series_equal(
        result.net_returns, result.gross_returns - result.costs, check_names=False
    )
    assert (result.costs >= 0).all()


def test_turnover_cannot_exceed_the_structural_limit(scores_frame, tradeable_frame):
    """One-way turnover is bounded by drop_n / top_k plus drift."""
    cfg = Config().replace(portfolio={"top_k": 13, "drop_n": 5,
                                      "cost_bps": 0.0, "slippage_bps": 0.0})
    weights = build_topk_dropn_weights(scores_frame, tradeable_frame, cfg)
    returns = pd.DataFrame(0.0, index=scores_frame.index, columns=scores_frame.columns)
    result = run_portfolio(weights, returns, cfg)
    steady = result.turnover.iloc[4:]
    assert steady.max() <= 5 / 13 + 1e-9
