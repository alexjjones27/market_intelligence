"""Assemble the results report from the backtest artefacts.

Kept separate from the backtest so the write-up can be regenerated without
re-running an hour of model fitting, and so the numbers in the prose always come
from the CSVs rather than from anything retyped by hand.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPORTS = Path("reports")


def _pct(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _num(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


LABELS = {
    "full": "Full pipeline (CSA + RPA)",
    "no_csa": "Ablation: no Confidence agent",
    "no_rpa": "Ablation: no Risk agent",
    "single_alpha": "Baseline: best single alpha",
    "gbm": "Baseline: gradient boosting",
    "no_agents_all_alphas": "Baseline: MLP on all alphas (no agents)",
    "sp500_cap_weight": "Benchmark: S&P 500 cap-weight (SPY, total return)",
    "sp500_equal_weight": "Benchmark: S&P 500 equal-weight (RSP)",
    "sp500_price_index": "Benchmark: S&P 500 price index (^GSPC)",
    "universe_equal_weight": "Benchmark: equal-weight tradeable universe",
}


def overall_table(summary: pd.DataFrame) -> str:
    order = [k for k in LABELS if k in set(summary["label"])]
    rows = ["| Strategy | Cum. return (net) | Ann. return | Ann. vol | Sharpe | Sortino | Calmar | Max DD |",
            "|---|---|---|---|---|---|---|---|"]
    for key in order:
        r = summary[summary["label"] == key].iloc[0]
        rows.append(
            f"| {LABELS[key]} | {_pct(r['cum_return'])} | {_pct(r['ann_return'])} | "
            f"{_pct(r['ann_volatility'])} | {_num(r['sharpe'])} | {_num(r['sortino'])} | "
            f"{_num(r['calmar'])} | {_pct(r['max_drawdown'])} |"
        )
    return "\n".join(rows)


def gross_net_table(summary: pd.DataFrame) -> str:
    strategies = summary[summary["kind"] == "strategy"]
    rows = ["| Strategy | Cum. return gross | Cum. return net | Sharpe gross | Sharpe net | Ann. turnover | Ann. cost drag |",
            "|---|---|---|---|---|---|---|"]
    for _, r in strategies.iterrows():
        rows.append(
            f"| {LABELS.get(r['label'], r['label'])} | {_pct(r['cum_return_gross'])} | "
            f"{_pct(r['cum_return'])} | {_num(r['sharpe_gross'])} | {_num(r['sharpe'])} | "
            f"{_num(r.get('ann_turnover', np.nan), 1)}x | {_pct(r.get('ann_cost_drag', np.nan))} |"
        )
    return "\n".join(rows)


def fold_table(folds: pd.DataFrame, variant: str = "full") -> str:
    sub = folds[folds["variant"] == variant].sort_values("test_window")
    rows = ["| Test window | Alphas | Cum. return (net) | Ann. return | Sharpe (net) | Sharpe (gross) | Max DD | Turnover/day |",
            "|---|---|---|---|---|---|---|---|"]
    for _, r in sub.iterrows():
        if pd.isna(r.get("cum_return_net")):
            rows.append(f"| {r['test_window']} | - | - | - | - | - | - | {r.get('note', '')} |")
            continue
        rows.append(
            f"| {r['test_window']} | {int(r['n_selected'])} | {_pct(r['cum_return_net'])} | "
            f"{_pct(r['ann_return_net'])} | {_num(r['sharpe_net'])} | {_num(r['sharpe_gross'])} | "
            f"{_pct(r['max_dd_net'])} | {_pct(r['mean_daily_turnover'], 1)} |"
        )
    return "\n".join(rows)


def significance_table(sig: pd.DataFrame) -> str:
    rows = ["| Strategy | Mean daily return | HAC t-stat | HAC p | Sharpe | Bootstrap 95% CI | Excess vs EW universe (t) |",
            "|---|---|---|---|---|---|---|"]
    for _, r in sig.iterrows():
        ci = f"[{_num(r['sharpe_ci_low'])}, {_num(r['sharpe_ci_high'])}]"
        excess_t = r.get("excess_vs_universe_equal_weight_t", np.nan)
        rows.append(
            f"| {LABELS.get(r['strategy'], r['strategy'])} | {_pct(r['mean_daily_return'], 4)} | "
            f"{_num(r['hac_t_stat'])} | {_num(r['hac_p_value'], 3)} | {_num(r['sharpe_point'])} | "
            f"{ci} | {_num(excess_t)} |"
        )
    return "\n".join(rows)


def attribution_table(attrib: pd.DataFrame) -> str:
    rows = ["| Strategy | CAPM alpha (ann.) | CAPM t | Market beta | + Sector alpha (ann.) | Sector-model t | Model R² |",
            "|---|---|---|---|---|---|---|"]
    for _, r in attrib.iterrows():
        rows.append(
            f"| {LABELS.get(r['strategy'], r['strategy'])} | {_pct(r['capm_alpha_annual'])} | "
            f"{_num(r['capm_alpha_t'])} | {_num(r['capm_beta'])} | {_pct(r['sector_alpha_annual'])} | "
            f"{_num(r['sector_alpha_t'])} | {_num(r['sector_model_r2'])} |"
        )
    return "\n".join(rows)


def sweep_table(sweep: pd.DataFrame) -> str:
    rows = ["| w_confidence | w_risk | Cum. return (net) | Ann. return | Sharpe (net) | Max DD |",
            "|---|---|---|---|---|---|"]
    for _, r in sweep.iterrows():
        marker = " **(paper's choice)**" if abs(r["w_confidence"] - 0.6) < 1e-9 else ""
        rows.append(
            f"| {r['w_confidence']:.1f}{marker} | {r['w_risk']:.1f} | {_pct(r['cum_return'])} | "
            f"{_pct(r['ann_return'])} | {_num(r['sharpe'])} | {_pct(r['max_drawdown'])} |"
        )
    return "\n".join(rows)


def selection_frequency(selections: pd.DataFrame, variant: str = "full") -> str:
    sub = selections[selections["variant"] == variant]
    if sub.empty:
        return "_no selections recorded_"
    counts = (
        sub.groupby(["alpha", "category"])
        .agg(folds=("fold", "nunique"), mean_ic=("mean_ic", "mean"))
        .reset_index()
        .sort_values(["folds", "alpha"], ascending=[False, True])
    )
    rows = ["| Alpha | Category | Folds selected | Mean IC at selection |", "|---|---|---|---|"]
    for _, r in counts.iterrows():
        rows.append(f"| `{r['alpha']}` | {r['category']} | {int(r['folds'])} | {_num(r['mean_ic'], 4)} |")
    return "\n".join(rows)
