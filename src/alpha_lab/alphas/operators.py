"""Time-series and cross-section operators used by the Appendix A.3 alphas.

Everything here takes and returns a wide ``DataFrame`` (index = trading date,
columns = ticker) so alphas read close to the paper's own notation:
``DELAY(CLOSE, 14)``, ``SMA(CLOSE, 20)``, ``MAX(HIGH, 20)``.

Two invariants hold for every operator in this module, and the test suite
enforces both:

1. **No look-ahead.** Every windowed operation is backward-looking. Rolling
   windows are right-closed with ``min_periods == window``, so a value at date
   ``t`` is a function of ``t`` and earlier only, and warm-up periods are NaN
   rather than being computed off a partial window. ``ewm`` uses
   ``adjust=False``, which is the recursive (causal) form.
2. **No cross-contamination between names.** Operators act column-wise, so one
   ticker's history never leaks into another's factor value.

Wilder-style indicators (RSI, ATR, ADX) use ``alpha = 1/n`` exponential
smoothing, which is Wilder's own recursion. It differs marginally from the
SMA-seeded variant in some charting packages for the first few dozen bars;
the difference washes out well inside the burn-in period we discard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Frame = pd.DataFrame

# ---------------------------------------------------------------- primitives


def safe_divide(numerator: Frame, denominator: Frame, eps: float = 1e-12) -> Frame:
    """Element-wise division that yields NaN rather than +/-inf on a zero
    denominator. An inf would survive every downstream mean and quietly poison
    a whole cross-section."""
    denom = denominator.where(denominator.abs() > eps)
    return numerator / denom


def DELAY(x: Frame, n: int) -> Frame:
    """Value n periods ago. Positive n only -- a negative shift is look-ahead."""
    if n < 0:
        raise ValueError(f"DELAY requires n >= 0, got {n}")
    return x.shift(n)


def DELTA(x: Frame, n: int) -> Frame:
    return x - x.shift(n)


def SMA(x: Frame, n: int) -> Frame:
    return x.rolling(n, min_periods=n).mean()


MEAN = SMA
MA = SMA


def EMA(x: Frame, n: int) -> Frame:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def WILDER(x: Frame, n: int) -> Frame:
    """Wilder's smoothing: alpha = 1/n rather than the 2/(n+1) of a span-n EMA."""
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def STD(x: Frame, n: int) -> Frame:
    return x.rolling(n, min_periods=n).std()


def VAR(x: Frame, n: int) -> Frame:
    return x.rolling(n, min_periods=n).var()


def MAX(x: Frame, n: int) -> Frame:
    return x.rolling(n, min_periods=n).max()


def MIN(x: Frame, n: int) -> Frame:
    return x.rolling(n, min_periods=n).min()


def SUM(x: Frame, n: int) -> Frame:
    return x.rolling(n, min_periods=n).sum()


def MEAN_DEV(x: Frame, n: int) -> Frame:
    """Rolling mean absolute deviation about the rolling mean (CCI's denominator)."""
    return x.rolling(n, min_periods=n).apply(
        lambda w: np.abs(w - w.mean()).mean(), raw=True
    )


def LOG(x: Frame) -> Frame:
    """Natural log, NaN for non-positive inputs (a price of 0 is bad data)."""
    return np.log(x.where(x > 0))


def SQRT(x: Frame) -> Frame:
    return np.sqrt(x.where(x >= 0))


def ABS(x: Frame) -> Frame:
    return x.abs()


def SIGN(x: Frame) -> Frame:
    return np.sign(x)


def IF(condition: Frame, then: Frame, otherwise: Frame | float) -> Frame:
    """Vectorised conditional matching the paper's ``IF(cond, a, b)``."""
    return then.where(condition, otherwise)


def RETURNS(close: Frame, n: int = 1) -> Frame:
    return close.pct_change(n, fill_method=None)


def LOG_RETURNS(close: Frame, n: int = 1) -> Frame:
    return LOG(close) - LOG(close).shift(n)


# ------------------------------------------------------- technical indicators


def TYPICAL_PRICE(high: Frame, low: Frame, close: Frame) -> Frame:
    return (high + low + close) / 3.0


def TRUE_RANGE(high: Frame, low: Frame, close: Frame) -> Frame:
    """Wilder's true range.

    ``np.fmax`` rather than ``np.maximum`` so the first bar of each series --
    which has no prior close, making two of the three candidates NaN -- falls
    back to ``high - low``, the standard convention. The result is still masked
    wherever the bar itself is missing, so a gap in the price series stays a
    gap instead of being papered over.
    """
    prev_close = close.shift(1)
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    tr = a.combine(b, np.fmax).combine(c, np.fmax)
    return tr.where(high.notna() & low.notna())


def ATR(high: Frame, low: Frame, close: Frame, n: int = 14) -> Frame:
    return WILDER(TRUE_RANGE(high, low, close), n)


def RSI(close: Frame, n: int = 14) -> Frame:
    """Wilder's RSI. Returns 100 where there are no losses in the window."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = WILDER(gain, n)
    avg_loss = WILDER(loss, n)
    rs = safe_divide(avg_gain, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 means an unbroken advance: RSI is 100 by definition, but
    # safe_divide has turned it into NaN. Restore it where gains exist.
    all_gain = (avg_loss.abs() <= 1e-12) & (avg_gain > 0)
    return rsi.mask(all_gain, 100.0)


def MACD(close: Frame, fast: int = 12, slow: int = 26) -> Frame:
    return EMA(close, fast) - EMA(close, slow)


def MACD_SIGNAL(close: Frame, fast: int = 12, slow: int = 26, signal: int = 9) -> Frame:
    return EMA(MACD(close, fast, slow), signal)


def ADX(high: Frame, low: Frame, close: Frame, n: int = 14) -> Frame:
    """Wilder's Average Directional Index (trend strength, 0-100)."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = WILDER(TRUE_RANGE(high, low, close), n)
    plus_di = 100.0 * safe_divide(WILDER(plus_dm, n), atr)
    minus_di = 100.0 * safe_divide(WILDER(minus_dm, n), atr)
    dx = 100.0 * safe_divide((plus_di - minus_di).abs(), plus_di + minus_di)
    return WILDER(dx, n)


def BOLLINGER(close: Frame, n: int = 20, k: float = 2.0) -> tuple[Frame, Frame]:
    """(upper, lower) Bollinger bands. A.3 does not state ``k``; 2.0 is the
    universal convention and is exposed here so it can be swept."""
    mid = SMA(close, n)
    width = k * STD(close, n)
    return mid + width, mid - width


def KELTNER(
    high: Frame, low: Frame, close: Frame, n: int = 20, atr_n: int = 10, mult: float = 2.0
) -> tuple[Frame, Frame]:
    """(upper, lower) Keltner channels.

    A.3 names the channel but specifies neither the centre line nor the
    multiplier. We use Chester Keltner's modern convention: an EMA centre with
    ATR-scaled bands. Documented rather than assumed, because a different
    convention shifts the resulting alpha.
    """
    mid = EMA(close, n)
    band = mult * ATR(high, low, close, atr_n)
    return mid + band, mid - band


def STOCHASTIC(high: Frame, low: Frame, close: Frame, n: int = 14) -> Frame:
    hh, ll = MAX(high, n), MIN(low, n)
    return 100.0 * safe_divide(close - ll, hh - ll)


def WILLIAMS_R(high: Frame, low: Frame, close: Frame, n: int = 14) -> Frame:
    hh, ll = MAX(high, n), MIN(low, n)
    return -100.0 * safe_divide(hh - close, hh - ll)


def CCI(high: Frame, low: Frame, close: Frame, n: int = 20) -> Frame:
    tp = TYPICAL_PRICE(high, low, close)
    return safe_divide(tp - SMA(tp, n), 0.015 * MEAN_DEV(tp, n))


def OBV(close: Frame, volume: Frame) -> Frame:
    """On-Balance Volume: running signed-volume total.

    A.3 writes an unbounded ``SUM(VOLUME * SIGN(...))``. An unbounded cumulative
    sum is not comparable across stocks with different listing histories, but it
    *is* what the paper specifies, so that is what this returns; the
    cross-sectional normalisation in Stage 1 is what makes it usable.
    """
    direction = SIGN(close.diff())
    return (volume * direction).cumsum()


def DRAWDOWN(close: Frame, n: int = 14) -> Frame:
    """Percent drawdown from the trailing n-day running maximum (<= 0)."""
    peak = MAX(close, n)
    return 100.0 * safe_divide(close - peak, peak)


# ------------------------------------------------------- volatility estimators


def GARMAN_KLASS(open_: Frame, high: Frame, low: Frame, close: Frame) -> Frame:
    """Single-bar Garman-Klass variance, square-rooted, exactly as A.3 writes it.

    Note this is a *one-day* estimator: the paper's formula has no window. It is
    correspondingly noisy -- the usual practice is to average the variance over
    20 days before taking the root. Implemented as specified; the deviation from
    convention is called out in the report rather than silently corrected.
    """
    hl = LOG(safe_divide(high, low))
    co = LOG(safe_divide(close, open_))
    var = 0.5 * hl**2 - (2.0 * np.log(2.0) - 1.0) * co**2
    return SQRT(var)


def PARKINSON(high: Frame, low: Frame, n: int = 20) -> Frame:
    hl = LOG(safe_divide(high, low))
    return SQRT(SUM(hl**2, n) / (4.0 * n * np.log(2.0)))


def YANG_ZHANG(open_: Frame, high: Frame, low: Frame, close: Frame, n: int = 20) -> Frame:
    """Yang-Zhang-style volatility, transcribed from A.3's formula.

    A.3 writes ``SQRT(VAR(LOG(C/O)) + 0.5*VAR(LOG(H/O) - LOG(L/O))
    + 0.25*VAR(LOG(C/DELAY(O,1))))``, which is *not* the canonical Yang-Zhang
    estimator (that one combines overnight variance, open-to-close variance and
    a Rogers-Satchell term with a k-weight derived from n). We implement the
    paper's version, with a 20-day window supplied for the unspecified VAR
    horizon, and flag the divergence.
    """
    co = LOG(safe_divide(close, open_))
    ho = LOG(safe_divide(high, open_))
    lo = LOG(safe_divide(low, open_))
    c_prev_o = LOG(safe_divide(close, open_.shift(1)))
    return SQRT(VAR(co, n) + 0.5 * VAR(ho - lo, n) + 0.25 * VAR(c_prev_o, n))


def ULCER_INDEX(close: Frame, n: int = 14) -> Frame:
    dd = DRAWDOWN(close, n)
    return SQRT(SMA(dd**2, n))


# ------------------------------------------------------- cross-section helpers


def cs_rank(x: Frame) -> Frame:
    """Per-day cross-sectional rank, scaled to [0, 1]."""
    return x.rank(axis=1, pct=True)


def cs_zscore(x: Frame, min_count: int = 20) -> Frame:
    """Per-day cross-sectional z-score.

    Uses only same-day values, so it cannot leak across time. Days with fewer
    than ``min_count`` observations return NaN -- a z-score over five names is
    noise dressed as a signal.
    """
    counts = x.notna().sum(axis=1)
    mu = x.mean(axis=1)
    sigma = x.std(axis=1)
    z = x.sub(mu, axis=0).div(sigma.where(sigma > 1e-12), axis=0)
    return z.where(counts >= min_count, np.nan)


def cs_winsorize(x: Frame, quantile: float = 0.01) -> Frame:
    """Clip each day's cross-section to its [q, 1-q] quantiles."""
    if quantile <= 0:
        return x
    lower = x.quantile(quantile, axis=1)
    upper = x.quantile(1.0 - quantile, axis=1)
    return x.clip(lower=lower, upper=upper, axis=0)
