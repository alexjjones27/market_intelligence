"""Significance testing and risk-factor attribution.

A point estimate of a Sharpe ratio from one backtest is not evidence. Three
things are needed before an apparent edge is worth believing, and all three are
implemented here:

1. **A standard error that accounts for autocorrelation.** Daily strategy
   returns are serially correlated (overlapping signals, momentum in the book),
   so the ordinary t-statistic on a mean return is too large. Newey-West/HAC
   standard errors correct for it.
2. **A distribution, not just a mean.** A stationary block bootstrap resamples
   *blocks* of consecutive days, preserving the autocorrelation the iid
   bootstrap would destroy, and yields a confidence interval for the Sharpe and
   for the excess return over a benchmark.
3. **Attribution against boring explanations.** An apparent alpha is very often
   a sector tilt or a beta tilt in disguise. :func:`factor_attribution` regresses
   strategy returns on the market plus equal-weight sector portfolios and reports
   the intercept with HAC errors: that intercept, not the raw return, is what
   survives as unexplained.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class HACResult:
    mean: float
    t_stat: float
    p_value: float
    std_error: float
    n_obs: int
    lags: int

    @property
    def significant_5pct(self) -> bool:
        return bool(np.isfinite(self.p_value) and self.p_value < 0.05)


def _default_lags(n: int) -> int:
    """Newey-West lag selection, floor(4 * (n/100)^(2/9)), the usual rule."""
    return max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


def hac_mean_test(returns: pd.Series, lags: int | None = None) -> HACResult:
    """Is the mean daily return distinguishable from zero, HAC-corrected?"""
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 30:
        return HACResult(np.nan, np.nan, np.nan, np.nan, len(r), 0)
    lags = lags or _default_lags(len(r))
    model = sm.OLS(r.to_numpy(), np.ones((len(r), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    return HACResult(
        mean=float(model.params[0]),
        t_stat=float(model.tvalues[0]),
        p_value=float(model.pvalues[0]),
        std_error=float(model.bse[0]),
        n_obs=len(r),
        lags=lags,
    )


def hac_excess_test(
    strategy: pd.Series, benchmark: pd.Series, lags: int | None = None
) -> HACResult:
    """Same test on the strategy-minus-benchmark difference."""
    aligned = pd.concat([strategy, benchmark], axis=1).dropna()
    if aligned.empty:
        return HACResult(np.nan, np.nan, np.nan, np.nan, 0, 0)
    return hac_mean_test(aligned.iloc[:, 0] - aligned.iloc[:, 1], lags)


def block_bootstrap(
    returns: pd.Series,
    statistic,
    iterations: int = 2000,
    block_size: int = 21,
    seed: int = 17,
) -> dict:
    """Bootstrap a statistic by resampling contiguous blocks of days.

    Blocks rather than individual days because daily strategy returns are
    autocorrelated; an iid bootstrap would break that dependence and report a
    confidence interval that is too narrow, which is the failure mode that makes
    a marginal edge look decisive.
    """
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(r)
    if n < 2 * block_size:
        return {"point": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_value_gt_zero": np.nan, "iterations": 0}

    rng = np.random.default_rng(seed)
    values = r.to_numpy()
    n_blocks = int(np.ceil(n / block_size))
    samples = np.empty(iterations)
    for i in range(iterations):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        drawn = np.concatenate([values[s : s + block_size] for s in starts])[:n]
        samples[i] = statistic(pd.Series(drawn))

    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return {"point": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_value_gt_zero": np.nan, "iterations": 0}
    return {
        "point": float(statistic(r)),
        "ci_low": float(np.percentile(finite, 2.5)),
        "ci_high": float(np.percentile(finite, 97.5)),
        "p_value_gt_zero": float((finite <= 0).mean()),
        "iterations": int(finite.size),
    }


@dataclass(frozen=True)
class Attribution:
    alpha_daily: float
    alpha_annual: float
    alpha_t_stat: float
    alpha_p_value: float
    betas: pd.Series
    r_squared: float
    n_obs: int

    @property
    def alpha_survives_5pct(self) -> bool:
        return bool(np.isfinite(self.alpha_p_value) and self.alpha_p_value < 0.05)


def sector_portfolios(
    returns: pd.DataFrame, sectors: pd.Series, tradeable: pd.DataFrame
) -> pd.DataFrame:
    """Daily equal-weight return of each GICS sector, tradeable names only."""
    masked = returns.where(tradeable)
    out = {}
    for sector, group in sectors.dropna().groupby(sectors.dropna()):
        cols = [c for c in group.index if c in masked.columns]
        if len(cols) >= 3:
            out[f"sector_{sector.replace(' ', '_')}"] = masked[cols].mean(axis=1)
    return pd.DataFrame(out, index=returns.index)


def factor_attribution(
    strategy: pd.Series,
    market: pd.Series,
    sectors: pd.DataFrame | None = None,
    trading_days: int = 252,
    lags: int | None = None,
) -> Attribution:
    """Regress strategy returns on market and sector factors; report the intercept.

    The intercept is the part of the return that market beta and sector tilts do
    not explain. If it collapses once sectors are included, the 'alpha' was a
    sector bet -- exactly the contamination worth ruling out before believing a
    factor edge.
    """
    frame = pd.DataFrame({"y": strategy, "market": market})
    if sectors is not None and not sectors.empty:
        frame = frame.join(sectors, how="inner")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 60:
        return Attribution(np.nan, np.nan, np.nan, np.nan, pd.Series(dtype=float), np.nan, len(frame))

    y = frame["y"]
    x = sm.add_constant(frame.drop(columns=["y"]))
    lags = lags or _default_lags(len(frame))
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    alpha_daily = float(model.params["const"])
    return Attribution(
        alpha_daily=alpha_daily,
        alpha_annual=float((1.0 + alpha_daily) ** trading_days - 1.0),
        alpha_t_stat=float(model.tvalues["const"]),
        alpha_p_value=float(model.pvalues["const"]),
        betas=model.params.drop("const"),
        r_squared=float(model.rsquared),
        n_obs=len(frame),
    )
