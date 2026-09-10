"""Split-basis alignment between prices and as-filed per-share fundamentals.

Yahoo restates prices for splits retroactively; SEC XBRL does not restate
filings. Mixing the two silently corrupts every valuation ratio -- Apple's
FY2019 diluted EPS of $11.89 against a split-adjusted 2019 price near $50 reads
as a P/E of 4. These tests pin the correction in both directions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.data.fundamentals import _apply_split_basis


@pytest.fixture
def four_for_one() -> pd.DataFrame:
    """A 4:1 split midway through the window, in Yahoo's factor convention."""
    days = pd.bdate_range("2019-01-01", "2021-12-31")
    split_date = pd.Timestamp("2020-08-31")
    cum = pd.Series(np.where(days >= split_date, 4.0, 1.0), index=days)
    return pd.DataFrame({"AAPL": cum / cum.iloc[-1]})


def test_per_share_value_filed_before_a_split_is_scaled_down(four_for_one):
    """EPS filed pre-split must be divided by the split ratio to match prices."""
    filed = pd.Series([11.89], index=[pd.Timestamp("2019-10-31")])
    out = _apply_split_basis(filed, "eps", "AAPL", four_for_one)
    assert out.iloc[0] == pytest.approx(11.89 / 4)


def test_per_share_value_filed_after_a_split_is_untouched(four_for_one):
    filed = pd.Series([3.28], index=[pd.Timestamp("2020-10-30")])
    out = _apply_split_basis(filed, "eps", "AAPL", four_for_one)
    assert out.iloc[0] == pytest.approx(3.28)


def test_share_counts_move_the_opposite_way(four_for_one):
    """A pre-split share count must be multiplied up, not down."""
    filed = pd.Series([4.4e9], index=[pd.Timestamp("2019-10-31")])
    out = _apply_split_basis(filed, "shares_outstanding", "AAPL", four_for_one)
    assert out.iloc[0] == pytest.approx(4.4e9 * 4)


def test_market_cap_is_invariant_to_the_split(four_for_one):
    """The arithmetic check that the convention is self-consistent:
    adjusted price x adjusted share count must equal the true market cap."""
    factor_at_filing = four_for_one["AAPL"].asof(pd.Timestamp("2019-10-31"))
    raw_price, raw_shares = 248.76, 4.4e9
    adjusted_price = raw_price * factor_at_filing
    adjusted_shares = _apply_split_basis(
        pd.Series([raw_shares], index=[pd.Timestamp("2019-10-31")]),
        "shares_outstanding", "AAPL", four_for_one,
    ).iloc[0]
    assert adjusted_price * adjusted_shares == pytest.approx(raw_price * raw_shares, rel=1e-9)


def test_company_level_totals_are_left_alone(four_for_one):
    """Revenue and equity are not per-share, so a split must not touch them."""
    filed = pd.Series([2.6e11], index=[pd.Timestamp("2019-10-31")])
    for field in ("revenue", "total_equity", "net_income"):
        out = _apply_split_basis(filed, field, "AAPL", four_for_one)
        assert out.iloc[0] == pytest.approx(2.6e11)


def test_reverse_splits_are_handled(four_for_one):
    """A 1:8 reverse split scales the ratio the other way."""
    days = four_for_one.index
    cum = pd.Series(np.where(days >= pd.Timestamp("2021-07-30"), 0.125, 1.0), index=days)
    factors = pd.DataFrame({"GE": cum / cum.iloc[-1]})
    filed = pd.Series([1.0], index=[pd.Timestamp("2020-01-31")])
    out = _apply_split_basis(filed, "eps", "GE", factors)
    assert out.iloc[0] == pytest.approx(8.0)


def test_missing_split_data_is_a_no_op():
    filed = pd.Series([5.0], index=[pd.Timestamp("2020-01-31")])
    assert _apply_split_basis(filed, "eps", "XYZ", None).iloc[0] == pytest.approx(5.0)
    empty = pd.DataFrame(index=pd.bdate_range("2020-01-01", periods=5))
    assert _apply_split_basis(filed, "eps", "XYZ", empty).iloc[0] == pytest.approx(5.0)
