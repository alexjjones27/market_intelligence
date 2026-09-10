"""Walk-forward orchestration of the full three-stage pipeline.

Per fold, in strict order:

1. **Select** (Stage 2) as of the *last day of validation*. The agents see only
   IC observations whose forward returns had completed by then.
2. **Fit** (Stage 3) the weight-optimisation network on training rows, early
   stopping on validation rows. Training rows are trimmed so every row's forward
   return completed inside the training window -- a row three days before the
   train/validation boundary carries a five-day forward return that reaches into
   validation, and keeping it would leak.
3. **Predict** on the test window and build the top-k/drop-n book.

Nothing from the test window touches selection, fitting, or standardisation.
The same skeleton runs the ablations and baselines so every variant faces
identical folds, identical costs and an identical portfolio rule -- otherwise a
comparison measures the harness rather than the strategy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha_lab.agents.factor_stats import FactorStats, forward_return
from alpha_lab.agents.selection import Selection, select_alphas
from alpha_lab.alphas.registry import AlphaPanel
from alpha_lab.backtest import baselines
from alpha_lab.backtest.walkforward import Fold
from alpha_lab.config import Config
from alpha_lab.data.panel import MarketPanel
from alpha_lab.models.mlp import TrainedModel, train_weight_optimizer
from alpha_lab.portfolio.construction import BacktestResult, build_topk_dropn_weights, run_portfolio

logger = logging.getLogger(__name__)


@dataclass
class FoldRun:
    fold: Fold
    variant: str
    result: BacktestResult | None
    selection: Selection | None = None
    model: TrainedModel | None = None
    notes: str = ""
    extras: dict = field(default_factory=dict)


def _window_dates(dates: pd.DatetimeIndex, start, end) -> pd.DatetimeIndex:
    return dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


def _trim_for_horizon(dates: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """Drop the last ``horizon`` sessions so each row's forward return completed
    inside the window it is being used to fit on."""
    return dates[:-horizon] if len(dates) > horizon else dates[:0]


def build_feature_matrix(
    alpha_panel: AlphaPanel, names: list[str], dates: pd.DatetimeIndex
) -> pd.DataFrame:
    return alpha_panel.stack(names, dates)


def build_target(panel: MarketPanel, horizon: int, dates: pd.DatetimeIndex) -> pd.Series:
    fwd = forward_return(panel, horizon).where(panel.tradeable).loc[dates]
    target = fwd.stack(future_stack=True)
    target.index.names = ["date", "ticker"]
    return target


def _scores_from_predictions(predictions: pd.Series, dates: pd.DatetimeIndex,
                             tickers: list[str]) -> pd.DataFrame:
    wide = predictions.unstack("ticker")
    return wide.reindex(index=dates, columns=tickers)


def run_pipeline_fold(
    alpha_panel: AlphaPanel,
    stats: FactorStats,
    panel: MarketPanel,
    cfg: Config,
    fold: Fold,
    variant: str = "full",
) -> FoldRun:
    """Stages 2 and 3 end to end on one fold."""
    horizon = cfg.mlp.target_horizon
    dates = panel.dates

    train_dates = _trim_for_horizon(_window_dates(dates, fold.train_start, fold.train_end), horizon)
    val_dates = _trim_for_horizon(_window_dates(dates, fold.val_start, fold.val_end), horizon)
    test_dates = _window_dates(dates, fold.test_start, fold.test_end)

    if len(train_dates) < 60 or len(val_dates) < 20 or len(test_dates) < 10:
        return FoldRun(fold, variant, None, notes="insufficient dates in fold")

    selection = select_alphas(stats, alpha_panel.registry, cfg, fold.val_end)
    if not selection.alphas:
        return FoldRun(fold, variant, None, selection=selection,
                       notes="no alpha cleared the selection threshold")

    features_train = build_feature_matrix(alpha_panel, selection.alphas, train_dates)
    features_val = build_feature_matrix(alpha_panel, selection.alphas, val_dates)
    features_test = build_feature_matrix(alpha_panel, selection.alphas, test_dates)

    target_train = build_target(panel, horizon, train_dates).reindex(features_train.index)
    target_val = build_target(panel, horizon, val_dates).reindex(features_val.index)

    model = train_weight_optimizer(features_train, target_train, features_val, target_val, cfg.mlp)
    if model is None:
        return FoldRun(fold, variant, None, selection=selection,
                       notes="MLP could not be fitted (too few clean rows)")

    predictions = model.predict(features_test)
    scores = _scores_from_predictions(predictions, test_dates, panel.tickers)
    weights, trade_counts = build_topk_dropn_weights(
        scores, panel.tradeable.loc[test_dates], cfg
    )
    result = run_portfolio(weights, panel["returns"].loc[test_dates], cfg, trade_counts)

    return FoldRun(
        fold, variant, result, selection=selection, model=model,
        extras={
            "n_selected": len(selection.alphas),
            "selected": selection.alphas,
            "categories": selection.categories,
            "duplicate_collisions": selection.duplicate_collisions,
            "val_loss": model.val_loss,
            "epochs": model.epochs_run,
            "n_train_rows": int(features_train.dropna().shape[0]),
        },
    )


def run_all_alphas_fold(
    alpha_panel: AlphaPanel, stats: FactorStats, panel: MarketPanel, cfg: Config, fold: Fold
) -> FoldRun:
    """Stage 3 on every usable alpha, with the agent layer removed entirely."""
    horizon = cfg.mlp.target_horizon
    dates = panel.dates
    names = [a for a in alpha_panel.usable() if a in stats.ic.columns]
    if not names:
        return FoldRun(fold, "no_agents_all_alphas", None, notes="no usable alphas")

    train_dates = _trim_for_horizon(_window_dates(dates, fold.train_start, fold.train_end), horizon)
    val_dates = _trim_for_horizon(_window_dates(dates, fold.val_start, fold.val_end), horizon)
    test_dates = _window_dates(dates, fold.test_start, fold.test_end)
    if len(train_dates) < 60 or len(val_dates) < 20 or len(test_dates) < 10:
        return FoldRun(fold, "no_agents_all_alphas", None, notes="insufficient dates")

    f_train = build_feature_matrix(alpha_panel, names, train_dates)
    f_val = build_feature_matrix(alpha_panel, names, val_dates)
    f_test = build_feature_matrix(alpha_panel, names, test_dates)
    y_train = build_target(panel, horizon, train_dates).reindex(f_train.index)
    y_val = build_target(panel, horizon, val_dates).reindex(f_val.index)

    model = train_weight_optimizer(f_train, y_train, f_val, y_val, cfg.mlp)
    if model is None:
        return FoldRun(fold, "no_agents_all_alphas", None, notes="MLP could not be fitted")

    scores = _scores_from_predictions(model.predict(f_test), test_dates, panel.tickers)
    weights, trade_counts = build_topk_dropn_weights(
        scores, panel.tradeable.loc[test_dates], cfg
    )
    result = run_portfolio(weights, panel["returns"].loc[test_dates], cfg, trade_counts)
    return FoldRun(fold, "no_agents_all_alphas", result, model=model,
                   extras={"n_selected": len(names)})


def run_single_alpha_fold(
    alpha_panel: AlphaPanel, stats: FactorStats, panel: MarketPanel, cfg: Config, fold: Fold
) -> FoldRun:
    """Best single factor by observable IC, through the same portfolio rule."""
    name, sign = baselines.best_single_alpha(
        stats, fold.val_end,
        None if cfg.csa.window == "expanding" else cfg.csa.window_days,
        cfg.csa.min_periods,
    )
    if name is None:
        return FoldRun(fold, "single_alpha", None, notes="no alpha had enough IC history")

    test_dates = _window_dates(panel.dates, fold.test_start, fold.test_end)
    scores = alpha_panel.normalized[name].loc[test_dates] * sign
    weights, trade_counts = build_topk_dropn_weights(
        scores, panel.tradeable.loc[test_dates], cfg
    )
    result = run_portfolio(weights, panel["returns"].loc[test_dates], cfg, trade_counts)
    return FoldRun(fold, "single_alpha", result,
                   extras={"alpha": name, "sign": sign, "n_selected": 1})


def run_gbm_fold(
    alpha_panel: AlphaPanel, stats: FactorStats, panel: MarketPanel, cfg: Config, fold: Fold
) -> FoldRun:
    """Gradient boosting on the agent-selected features -- the paper's XGBoost row."""
    horizon = cfg.mlp.target_horizon
    dates = panel.dates
    selection = select_alphas(stats, alpha_panel.registry, cfg, fold.val_end)
    if not selection.alphas:
        return FoldRun(fold, "gbm", None, notes="no alpha selected")

    train_dates = _trim_for_horizon(_window_dates(dates, fold.train_start, fold.train_end), horizon)
    val_dates = _trim_for_horizon(_window_dates(dates, fold.val_start, fold.val_end), horizon)
    test_dates = _window_dates(dates, fold.test_start, fold.test_end)

    f_train = build_feature_matrix(alpha_panel, selection.alphas, train_dates)
    f_val = build_feature_matrix(alpha_panel, selection.alphas, val_dates)
    f_test = build_feature_matrix(alpha_panel, selection.alphas, test_dates)
    y_train = build_target(panel, horizon, train_dates).reindex(f_train.index)
    y_val = build_target(panel, horizon, val_dates).reindex(f_val.index)

    model = baselines.fit_gradient_boosting(f_train, y_train, f_val, y_val, cfg.mlp.seed)
    if model is None:
        return FoldRun(fold, "gbm", None, notes="GBM could not be fitted")

    predictions = baselines.predict_frame(model, f_test, selection.alphas)
    scores = _scores_from_predictions(predictions, test_dates, panel.tickers)
    weights, trade_counts = build_topk_dropn_weights(
        scores, panel.tradeable.loc[test_dates], cfg
    )
    result = run_portfolio(weights, panel["returns"].loc[test_dates], cfg, trade_counts)
    return FoldRun(fold, "gbm", result, selection=selection,
                   extras={"n_selected": len(selection.alphas)})


VARIANTS: dict[str, dict] = {
    # Ablation grid for the paper's Tables 7 and 8.
    "full": {"use_csa": True, "use_rpa": True},
    "no_csa": {"use_csa": False, "use_rpa": True},
    "no_rpa": {"use_csa": True, "use_rpa": False},
}


def run_walk_forward(
    alpha_panel: AlphaPanel,
    stats: FactorStats,
    panel: MarketPanel,
    cfg: Config,
    folds: list[Fold],
    variants: list[str] | None = None,
    include_baselines: bool = True,
) -> list[FoldRun]:
    """Every requested variant across every fold."""
    variants = variants or ["full"]
    runs: list[FoldRun] = []

    for fold in folds:
        for variant in variants:
            overrides = VARIANTS[variant]
            fold_cfg = cfg.replace(selection=overrides)
            run = run_pipeline_fold(alpha_panel, stats, panel, fold_cfg, fold, variant)
            runs.append(run)
            logger.info(
                "%s / %s: %s", fold.name, variant,
                run.notes or f"{run.extras.get('n_selected', 0)} alphas selected",
            )

        if include_baselines:
            runs.append(run_single_alpha_fold(alpha_panel, stats, panel, cfg, fold))
            runs.append(run_gbm_fold(alpha_panel, stats, panel, cfg, fold))
            runs.append(run_all_alphas_fold(alpha_panel, stats, panel, cfg, fold))

    return runs


def _stitch(runs: list[FoldRun], variant: str, attribute: str) -> pd.Series:
    """Concatenate one variant's out-of-sample test windows into one series.

    Folds do not overlap in test time (asserted in the walk-forward tests), so
    this is a genuine out-of-sample track record rather than a spliced-together
    in-sample one.
    """
    pieces = [
        getattr(r.result, attribute)
        for r in runs
        if r.variant == variant and r.result is not None
    ]
    if not pieces:
        return pd.Series(dtype=float)
    joined = pd.concat(pieces).sort_index()
    return joined[~joined.index.duplicated(keep="first")]


def stitch_returns(runs: list[FoldRun], variant: str, net: bool = True) -> pd.Series:
    return _stitch(runs, variant, "net_returns" if net else "gross_returns")


def stitch_turnover(runs: list[FoldRun], variant: str) -> pd.Series:
    return _stitch(runs, variant, "turnover")


def stitch_costs(runs: list[FoldRun], variant: str) -> pd.Series:
    return _stitch(runs, variant, "costs")
