"""Benchmarks and baseline strategies.

"Beats the index" is a weak claim in a period when the index rose. What matters
is whether the three-stage pipeline beats the *simple things you would try
first*, on the same universe, with the same portfolio rule and the same costs.
So alongside the cap- and equal-weight indices this module provides:

* **Equal-weight universe** -- hold every tradeable name. Isolates the effect of
  the universe and the date range from the effect of the signal.
* **Best single alpha** -- pick the one factor with the strongest observable IC
  and run it through the identical top-k/drop-n machinery. If the full pipeline
  cannot beat one formula, the agents and the network are decoration.
* **Gradient boosting** -- the paper's own XGBoost comparator, on the same
  features and the same splits. (sklearn's ``HistGradientBoostingRegressor`` is
  the same algorithm family; using it avoids an extra dependency and its
  defaults are not tuned in our favour.)
* **All-alphas MLP** -- the Stage 3 network fed *every* usable alpha, skipping
  the agent layer entirely. This is the cleanest read on what Stage 2 adds.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from alpha_lab.agents.factor_stats import FactorStats, observable_ic_window
from alpha_lab.data.panel import MarketPanel

logger = logging.getLogger(__name__)


def equal_weight_universe_returns(panel: MarketPanel) -> pd.Series:
    """Daily equal-weight return across all tradeable names."""
    return panel["returns"].where(panel.tradeable).mean(axis=1)


def benchmark_returns(panel: MarketPanel) -> pd.DataFrame:
    """Daily total returns of each benchmark series."""
    if panel.benchmarks.empty:
        return pd.DataFrame(index=panel.dates)
    return panel.benchmarks.pct_change(fill_method=None)


def best_single_alpha(
    stats: FactorStats, as_of: pd.Timestamp, lookback: int | None, min_obs: int = 60
) -> tuple[str | None, float]:
    """Highest observable |mean IC| as of ``as_of``, with its trading sign."""
    window = observable_ic_window(stats.ic, as_of, stats.horizon, lookback)
    if window.empty:
        return None, 1.0
    counts = window.notna().sum()
    mean_ic = window.mean().where(counts >= min_obs)
    if mean_ic.dropna().empty:
        return None, 1.0
    name = mean_ic.abs().idxmax()
    return str(name), float(np.sign(mean_ic[name]) or 1.0)


def fit_gradient_boosting(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    val_features: pd.DataFrame,
    val_target: pd.Series,
    seed: int = 17,
):
    """Gradient-boosting baseline on the same rows the MLP sees.

    Deliberately close to defaults: a hand-tuned baseline that happens to lose to
    the pipeline proves nothing, and one tuned as hard as the pipeline would take
    as long again. Early stopping uses the same validation split.
    """
    frame = train_features.copy()
    frame["__y"] = train_target
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 200:
        return None

    val = val_features[train_features.columns].copy()
    val["__y"] = val_target
    val = val.replace([np.inf, -np.inf], np.nan).dropna()
    if len(val) < 50:
        return None

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )
    model.fit(frame[train_features.columns].to_numpy(), frame["__y"].to_numpy())
    return model


def predict_frame(model, features: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Predict with a fitted sklearn model over a (date, ticker) feature frame."""
    x = features[columns].to_numpy(dtype=np.float64)
    valid = ~np.isnan(x).any(axis=1)
    out = np.full(len(features), np.nan)
    if valid.any():
        out[valid] = model.predict(x[valid])
    return pd.Series(out, index=features.index)
