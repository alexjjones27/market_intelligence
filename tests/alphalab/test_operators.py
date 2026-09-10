"""Operator correctness against hand-computed values.

These are deliberately arithmetic rather than property-based: an operator that
is subtly wrong (an off-by-one in DELAY, Wilder smoothing where a simple mean
belongs) still looks plausible on a chart, and only an explicit expected number
catches it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.alphas import operators as ops


def frame(values: list[float]) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.DataFrame({"A": values}, index=idx)


def test_delay_shifts_backwards_only():
    x = frame([1, 2, 3, 4, 5])
    assert ops.DELAY(x, 1)["A"].tolist()[1:] == [1.0, 2.0, 3.0, 4.0]
    assert np.isnan(ops.DELAY(x, 1)["A"].iloc[0])


def test_delay_rejects_negative_shift():
    """A negative DELAY is look-ahead; it must be impossible, not merely unused."""
    with pytest.raises(ValueError):
        ops.DELAY(frame([1, 2, 3]), -1)


def test_sma_requires_full_window():
    x = frame([1, 2, 3, 4, 5])
    sma = ops.SMA(x, 3)["A"]
    assert sma.isna().tolist()[:2] == [True, True]
    assert sma.iloc[2] == pytest.approx(2.0)
    assert sma.iloc[4] == pytest.approx(4.0)


def test_ema_is_recursive_and_causal():
    # adjust=False, min_periods=n: seed is the mean of the first n, then
    # EMA_t = a*x_t + (1-a)*EMA_{t-1} with a = 2/(n+1).
    x = frame([10, 11, 12, 13])
    ema = ops.EMA(x, 2)["A"]
    assert ema.isna().iloc[0]
    seed = ema.iloc[1]
    alpha = 2 / 3
    assert ema.iloc[2] == pytest.approx(alpha * 12 + (1 - alpha) * seed)


def test_wilder_uses_one_over_n():
    x = frame([10, 20, 30, 40])
    w = ops.WILDER(x, 2)["A"]
    seed = w.iloc[1]
    assert w.iloc[2] == pytest.approx(0.5 * 30 + 0.5 * seed)


def test_true_range_takes_the_widest_of_three():
    high = frame([10.0, 12.0])
    low = frame([9.0, 11.0])
    close = frame([9.5, 11.5])
    tr = ops.TRUE_RANGE(high, low, close)["A"]
    assert tr.iloc[0] == pytest.approx(1.0)          # no prior close: high - low
    # prior close 9.5 -> max(12-11, |12-9.5|, |11-9.5|) = 2.5
    assert tr.iloc[1] == pytest.approx(2.5)


def test_rsi_is_100_on_an_unbroken_advance():
    x = frame(list(range(1, 40)))
    assert ops.RSI(x, 14)["A"].iloc[-1] == pytest.approx(100.0)


def test_rsi_is_zero_on_an_unbroken_decline():
    x = frame(list(range(40, 1, -1)))
    assert ops.RSI(x, 14)["A"].iloc[-1] == pytest.approx(0.0)


def test_rsi_centres_on_fifty_when_gains_match_losses():
    """A perfectly alternating series has equal average gain and loss.

    Wilder smoothing is recursive, so the reading oscillates a couple of points
    either side of 50 depending on whether the latest bar was the up or the
    down leg. The invariant is the *average* of a full cycle, not any single
    bar -- asserting the latter would be asserting the oscillation phase.
    """
    x = frame([100 + (2 if i % 2 else -2) for i in range(200)])
    rsi = ops.RSI(x, 14)["A"]
    # RSI = 100*G/(G+L) is nonlinear in the smoothed averages, so the mean over
    # an up/down cycle sits near 50 rather than exactly on it. The invariants
    # worth asserting are that it stays tight around the midpoint and that it
    # actually alternates rather than drifting to an extreme.
    assert rsi.iloc[-2:].mean() == pytest.approx(50.0, abs=2.0)
    assert rsi.iloc[-20:].between(45, 55).all()
    assert rsi.iloc[-1] != pytest.approx(rsi.iloc[-2])


def test_stochastic_hits_the_band_edges():
    high = frame([10.0] * 20)
    low = frame([0.0] * 20)
    at_top = ops.STOCHASTIC(high, low, frame([10.0] * 20), 14)["A"].iloc[-1]
    at_bottom = ops.STOCHASTIC(high, low, frame([0.0] * 20), 14)["A"].iloc[-1]
    assert at_top == pytest.approx(100.0)
    assert at_bottom == pytest.approx(0.0)


def test_williams_r_is_the_stochastic_mirrored():
    n = 30
    rng = np.random.default_rng(0)
    close = frame(list(100 + rng.normal(0, 2, n)))
    high = close + 1.0
    low = close - 1.0
    k = ops.STOCHASTIC(high, low, close, 14)["A"]
    r = ops.WILLIAMS_R(high, low, close, 14)["A"]
    pd.testing.assert_series_equal(r, k - 100.0, check_names=False)


def test_safe_divide_returns_nan_not_inf():
    out = ops.safe_divide(frame([1.0, 2.0]), frame([0.0, 2.0]))["A"]
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(1.0)
    assert not np.isinf(out).any()


def test_log_rejects_non_positive_prices():
    out = ops.LOG(frame([1.0, 0.0, -5.0, np.e]))["A"]
    assert np.isnan(out.iloc[1]) and np.isnan(out.iloc[2])
    assert out.iloc[3] == pytest.approx(1.0)


def test_drawdown_is_zero_at_a_new_high_and_negative_below():
    x = frame([1, 2, 3, 4, 5, 4])
    dd = ops.DRAWDOWN(x, 3)["A"]
    assert dd.iloc[4] == pytest.approx(0.0)
    assert dd.iloc[5] == pytest.approx(-20.0)


def test_obv_accumulates_signed_volume():
    close = frame([10, 11, 10, 12])
    volume = frame([100, 200, 300, 400])
    obv = ops.OBV(close, volume)["A"]
    assert obv.iloc[1] == pytest.approx(200.0)
    assert obv.iloc[2] == pytest.approx(-100.0)
    assert obv.iloc[3] == pytest.approx(300.0)


def test_bollinger_bands_straddle_the_mean():
    rng = np.random.default_rng(3)
    x = frame(list(100 + rng.normal(0, 3, 60)))
    upper, lower = ops.BOLLINGER(x, 20, 2.0)
    mid = ops.SMA(x, 20)
    assert ((upper - mid) - (mid - lower)).abs().max().max() < 1e-9
    assert (upper.dropna() > lower.dropna()).all().all()


def test_cs_zscore_is_per_day_and_standardised():
    idx = pd.bdate_range("2020-01-01", periods=3)
    df = pd.DataFrame(
        np.array([[1.0, 2, 3, 4, 5], [10, 20, 30, 40, 50], [5, 5, 5, 5, 5]]),
        index=idx, columns=list("ABCDE"),
    )
    z = ops.cs_zscore(df, min_count=2)
    assert z.iloc[0].mean() == pytest.approx(0.0, abs=1e-12)
    assert z.iloc[0].std() == pytest.approx(1.0)
    # A day with the same value everywhere carries no cross-sectional signal.
    assert z.iloc[2].isna().all()


def test_cs_zscore_needs_a_minimum_cross_section():
    idx = pd.bdate_range("2020-01-01", periods=1)
    df = pd.DataFrame([[1.0, 2.0, 3.0]], index=idx, columns=list("ABC"))
    assert ops.cs_zscore(df, min_count=20).isna().all().all()


def test_cs_winsorize_clips_the_tails():
    idx = pd.bdate_range("2020-01-01", periods=1)
    df = pd.DataFrame([[-1000.0] + [1.0] * 98 + [1000.0]], index=idx)
    out = ops.cs_winsorize(df, 0.05)
    assert out.to_numpy().min() > -1000.0
    assert out.to_numpy().max() < 1000.0
