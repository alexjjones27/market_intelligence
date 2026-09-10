"""Walk-forward fold generation.

The paper reports a single Jan 2023 - Jan 2024 test window for its headline
53.17% result, and three fixed splits in Table 5. One window is not evidence:
with enough configurations, some window always looks spectacular. This module
rolls the paper's own train/validation/test geometry forward across the whole
sample so every fold is reported, not just the best one.

The default geometry reproduces Table 5's SP500 rows exactly on the first fold
(train Jan 2019-Jun 2020, validate Jun-Dec 2020, test Jan-Jun 2021) and then
steps forward six months at a time.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alpha_lab.config import Config


@dataclass(frozen=True)
class Fold:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def label(self) -> str:
        return f"{self.test_start.date()}..{self.test_end.date()}"

    def describe(self) -> dict:
        return {
            "fold": self.name,
            "train": f"{self.train_start.date()}..{self.train_end.date()}",
            "validation": f"{self.val_start.date()}..{self.val_end.date()}",
            "test": f"{self.test_start.date()}..{self.test_end.date()}",
        }


def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return ts + pd.offsets.MonthEnd(0)


def generate_folds(cfg: Config, available_end: pd.Timestamp | None = None) -> list[Fold]:
    """Rolling train/validation/test splits over the configured span."""
    wf = cfg.walkforward
    start = pd.Timestamp(wf.start)
    end = pd.Timestamp(wf.end)
    if available_end is not None:
        end = min(end, pd.Timestamp(available_end))

    folds: list[Fold] = []
    index = 0
    while True:
        train_start = start + pd.DateOffset(months=index * wf.step_months)
        train_end = _month_end(train_start + pd.DateOffset(months=wf.train_months) - pd.DateOffset(days=1))
        val_start = train_end + pd.Timedelta(days=1)
        val_end = _month_end(val_start + pd.DateOffset(months=wf.validation_months) - pd.DateOffset(days=1))
        test_start = val_end + pd.Timedelta(days=1)
        test_end = _month_end(test_start + pd.DateOffset(months=wf.test_months) - pd.DateOffset(days=1))

        if test_end > end:
            # Keep a final partial fold only if it has a meaningful test window.
            if test_start < end and (end - test_start).days >= 45:
                folds.append(
                    Fold(f"fold_{index + 1}", train_start, train_end, val_start, val_end,
                         test_start, min(test_end, end))
                )
            break
        folds.append(
            Fold(f"fold_{index + 1}", train_start, train_end, val_start, val_end,
                 test_start, test_end)
        )
        index += 1
        if index > 50:  # guard against a misconfigured step of zero
            break
    return folds


def paper_table5_folds() -> list[Fold]:
    """The three SP500 splits printed in the paper's Table 5, verbatim."""
    raw = [
        ("2019-01-01", "2020-06-30", "2020-06-01", "2020-12-31", "2021-01-01", "2021-06-30"),
        ("2020-01-01", "2021-06-30", "2021-06-01", "2021-12-31", "2022-01-01", "2022-06-30"),
        ("2021-01-01", "2022-06-30", "2022-06-01", "2022-12-31", "2023-01-01", "2023-06-30"),
    ]
    folds = []
    for i, (ts, te, vs, ve, xs, xe) in enumerate(raw, start=1):
        # Table 5 prints validation as starting in June while training runs to
        # the end of June, i.e. the windows overlap by a month as printed. We
        # start validation the day after training ends; an overlap would train
        # and early-stop on the same rows.
        train_end = pd.Timestamp(te)
        folds.append(
            Fold(
                f"paper_fold_{i}",
                pd.Timestamp(ts), train_end,
                train_end + pd.Timedelta(days=1), pd.Timestamp(ve),
                pd.Timestamp(xs), pd.Timestamp(xe),
            )
        )
    return folds
