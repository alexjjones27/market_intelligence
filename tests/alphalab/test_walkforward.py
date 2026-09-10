"""Fold geometry and the train/validation/test boundaries.

A walk-forward harness that silently overlaps its windows produces in-sample
results wearing an out-of-sample label, which is worse than no walk-forward at
all. These tests pin the boundaries, including the subtler one: training rows
must be trimmed by the forward-return horizon, because a row near the end of the
training window carries a target that reaches into validation.
"""
from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.backtest.engine import _trim_for_horizon, _window_dates
from alpha_lab.backtest.walkforward import generate_folds, paper_table5_folds
from alpha_lab.config import Config


@pytest.fixture
def folds():
    cfg = Config()
    return generate_folds(cfg, pd.Timestamp("2024-06-28"))


def test_windows_are_ordered_and_never_overlap(folds):
    for fold in folds:
        assert fold.train_start < fold.train_end < fold.val_start
        assert fold.val_start < fold.val_end < fold.test_start
        assert fold.test_start < fold.test_end


def test_test_windows_do_not_overlap_each_other(folds):
    """Stitching folds into one track record is only honest if the test windows
    are disjoint."""
    spans = sorted((f.test_start, f.test_end) for f in folds)
    for (_, end), (next_start, _) in zip(spans, spans[1:]):
        assert end < next_start


def test_folds_roll_forward_by_the_configured_step(folds):
    for a, b in zip(folds, folds[1:]):
        months = (b.train_start.year - a.train_start.year) * 12 + (
            b.train_start.month - a.train_start.month
        )
        assert months == Config().walkforward.step_months


def test_first_fold_reproduces_the_papers_table5_split(folds):
    """Table 5's first SP500 row: train Jan 2019-Jun 2020, validate to Dec 2020,
    test Jan-Jun 2021."""
    first = folds[0]
    assert first.train_start == pd.Timestamp("2019-01-01")
    assert first.train_end == pd.Timestamp("2020-06-30")
    assert first.test_start == pd.Timestamp("2021-01-01")
    assert first.test_end == pd.Timestamp("2021-06-30")


def test_paper_folds_do_not_overlap_train_and_validation():
    """Table 5 prints validation starting in June while training runs to the end
    of June. Training and early-stopping on the same rows would defeat the
    validation split, so we start validation the day after training ends."""
    for fold in paper_table5_folds():
        assert fold.val_start > fold.train_end


def test_training_rows_are_trimmed_by_the_forward_horizon():
    """The last `horizon` training sessions carry targets that land in
    validation, so they must be dropped from the fit."""
    dates = pd.bdate_range("2021-01-04", periods=100)
    window = _window_dates(dates, dates[0], dates[49])
    trimmed = _trim_for_horizon(window, horizon=5)
    assert len(trimmed) == len(window) - 5
    assert trimmed.max() == window[-6]


def test_trimming_a_window_shorter_than_the_horizon_yields_nothing():
    dates = pd.bdate_range("2021-01-04", periods=3)
    assert len(_trim_for_horizon(dates, horizon=5)) == 0


def test_trimmed_training_targets_cannot_reach_the_test_window(folds):
    """End to end: the newest training row's forward return must complete
    before the test window opens."""
    dates = pd.bdate_range("2019-01-01", "2024-06-28")
    horizon = Config().mlp.target_horizon
    for fold in folds:
        train = _trim_for_horizon(_window_dates(dates, fold.train_start, fold.train_end), horizon)
        if len(train) == 0:
            continue
        last_target_date = dates[dates.get_loc(train.max()) + horizon]
        assert last_target_date <= fold.train_end
        assert last_target_date < fold.test_start
