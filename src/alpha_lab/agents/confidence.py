"""Confidence Score Agent (CSA).

The paper defines confidence only as ``theta_ij = E[IC(alpha | market state)]``
(equation 5) and never says how the expectation is estimated. We operationalise
it as the **mean rolling out-of-sample rank IC**, computed strictly on IC
observations whose forward returns had already completed by the scoring date
(see ``factor_stats.observable_ic_window``).

One judgement call worth stating plainly: we score on the **magnitude** of the
mean IC, not its signed value. An alpha with IC = -0.05 is exactly as useful as
one with +0.05 -- you trade it the other way round -- and the paper itself
reports negative ICs throughout Table 3 while describing a combination IC of
-0.0587 as "quite high", which only makes sense on magnitude. The sign is
retained separately so downstream stages can orient each factor; the MLP learns
it as the weight's sign anyway. Set ``confidence_metric`` to ``mean_ic`` for the
signed reading, or ``ic_ir`` to score on the IC's t-statistic instead (mean over
standard deviation), which rewards consistency rather than raw size.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.agents.factor_stats import FactorStats, observable_ic_window
from alpha_lab.config import Config


@dataclass(frozen=True)
class ConfidenceScores:
    score: pd.Series
    """alpha -> confidence, higher is better."""

    sign: pd.Series
    """alpha -> +1/-1, the direction the factor should be traded."""

    n_obs: pd.Series
    raw_mean_ic: pd.Series


def score_confidence(
    stats: FactorStats, cfg: Config, as_of: pd.Timestamp
) -> ConfidenceScores:
    """Confidence for every alpha, as knowable on ``as_of``."""
    lookback = None if cfg.csa.window == "expanding" else cfg.csa.window_days
    window = observable_ic_window(stats.ic, as_of, stats.horizon, lookback)

    n_obs = window.notna().sum()
    mean_ic = window.mean()
    std_ic = window.std()

    metric = cfg.csa.confidence_metric
    if metric == "mean_ic":
        score = mean_ic
    elif metric == "ic_ir":
        score = (mean_ic / std_ic.where(std_ic > 0)).abs()
    else:
        score = mean_ic.abs()

    # An alpha with too little history is not "low confidence", it is unmeasured.
    # Dropping it to NaN keeps it out of selection instead of ranking it last.
    insufficient = n_obs < cfg.csa.min_periods
    score = score.where(~insufficient)

    sign = np.sign(mean_ic).replace(0.0, 1.0).where(~insufficient)
    return ConfidenceScores(
        score=score.astype(float),
        sign=sign.astype(float),
        n_obs=n_obs.astype(int),
        raw_mean_ic=mean_ic.astype(float),
    )
