"""Build reports/RESULTS.md from the backtest artefacts.

Every number in the prose is read from the CSVs rather than retyped, and the
verdict sentences are derived from the numbers so the write-up cannot drift
away from what the run actually produced.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpha_lab.reporting.report import (
    LABELS, attribution_table, fold_table, gross_net_table, overall_table,
    selection_frequency, significance_table, sweep_table, _num, _pct,
)

REPORTS = Path("reports")


def read(name: str) -> pd.DataFrame:
    path = REPORTS / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    meta = json.loads((REPORTS / "results_meta.json").read_text())
    summary = read("results_overall.csv")
    folds = read("results_by_fold.csv")
    sig = read("results_significance.csv")
    attrib = read("results_attribution.csv")
    sweep = read("results_weight_sweep.csv")
    selections = read("results_selections.csv")
    fold_windows = read("results_folds.csv")

    def stat(label: str, column: str) -> float:
        row = summary[summary["label"] == label]
        return float(row.iloc[0][column]) if len(row) else np.nan

    def sig_stat(label: str, column: str) -> float:
        row = sig[sig["strategy"] == label]
        return float(row.iloc[0][column]) if len(row) else np.nan

    full_net = stat("full", "cum_return")
    full_sharpe = stat("full", "sharpe")
    full_gross = stat("full", "cum_return_gross")
    ew = stat("universe_equal_weight", "cum_return")
    cap = stat("sp500_cap_weight", "cum_return")
    eq_bench = stat("sp500_equal_weight", "cum_return")
    single = stat("single_alpha", "cum_return")
    gbm = stat("gbm", "cum_return")
    all_alphas = stat("no_agents_all_alphas", "cum_return")
    no_csa = stat("no_csa", "cum_return")
    no_rpa = stat("no_rpa", "cum_return")

    hac_t = sig_stat("full", "hac_t_stat")
    hac_p = sig_stat("full", "hac_p_value")
    ci_low = sig_stat("full", "sharpe_ci_low")
    ci_high = sig_stat("full", "sharpe_ci_high")
    ex_t = sig_stat("full", "excess_vs_universe_equal_weight_t")
    ex_p = sig_stat("full", "excess_vs_universe_equal_weight_p")

    attrib_full = attrib[attrib["strategy"] == "full"]
    capm_alpha = float(attrib_full.iloc[0]["capm_alpha_annual"]) if len(attrib_full) else np.nan
    capm_t = float(attrib_full.iloc[0]["capm_alpha_t"]) if len(attrib_full) else np.nan
    sector_alpha = float(attrib_full.iloc[0]["sector_alpha_annual"]) if len(attrib_full) else np.nan
    sector_t = float(attrib_full.iloc[0]["sector_alpha_t"]) if len(attrib_full) else np.nan
    beta = float(attrib_full.iloc[0]["capm_beta"]) if len(attrib_full) else np.nan

    fold_full = folds[folds["variant"] == "full"].dropna(subset=["sharpe_net"])
    n_positive = int((fold_full["cum_return_net"] > 0).sum())
    n_folds = len(fold_full)
    sharpe_spread = (
        f"{_num(fold_full['sharpe_net'].min())} to {_num(fold_full['sharpe_net'].max())}"
        if n_folds else "n/a"
    )
    cost_drag = stat("full", "ann_cost_drag")

    L: list[str] = []
    add = L.append

    add("# Replication results: LLM-driven alpha framework on the S&P 500\n")
    add("Backtest of the three-stage framework from Kou et al., *Automate Strategy "
        "Finding with LLM in Quant Investment* (Findings of EMNLP 2025), adapted to "
        "US equities. Build details, data provenance and the leakage audit are in "
        "[ALPHA_LAB.md](../ALPHA_LAB.md).\n")

    add("## Run configuration\n")
    add(f"- Out-of-sample period: **{meta['oos_start']} to {meta['oos_end']}** "
        f"across **{meta['n_folds']} walk-forward folds**")
    add(f"- Universe: point-in-time S&P 500, **{meta['n_alphas_usable']} usable alphas** "
        f"of 92 in Appendix A.3")
    add(f"- Portfolio: top-{meta['top_k']} / drop-{meta['drop_n']}, equal weight, "
        f"**{meta['cost_bps_per_side']:.0f} bps per side** (cost + slippage), 1-day execution lag")
    add(f"- Agent weights: w_confidence **{meta['w_confidence']}**, w_risk **{meta['w_risk']}**; "
        f"IC horizon **{meta['ic_horizon']} days**\n")
    add("Every fold is reported. None is singled out.\n")

    add("### Fold geometry\n")
    if not fold_windows.empty:
        add("| Fold | Train | Validation | Test |")
        add("|---|---|---|---|")
        for _, r in fold_windows.iterrows():
            add(f"| {r['fold']} | {r['train']} | {r['validation']} | {r['test']} |")
    add("")

    add("## Headline: stitched out-of-sample performance, net of costs\n")
    add(overall_table(summary))
    add("")
    add(f"The full pipeline returned **{_pct(full_net)} cumulative** over "
        f"{meta['oos_start']} to {meta['oos_end']}, against **{_pct(ew)}** for an "
        f"equal-weight hold of the same tradeable universe and **{_pct(cap)}** for "
        f"cap-weight S&P 500 total return.\n")

    add("## Costs are not a rounding error\n")
    add(gross_net_table(summary))
    add("")
    add(f"The paper models no transaction costs at all. On this universe the "
        f"pipeline turns over roughly **{_pct(stat('full', 'mean_daily_turnover'), 1)} of the "
        f"book per day**, which at {meta['cost_bps_per_side']:.0f}bps per side costs "
        f"**{_pct(cost_drag)} a year**. Cumulative return falls from "
        f"{_pct(full_gross)} gross to {_pct(full_net)} net. Any comparison against the "
        f"paper's headline number has to account for this before anything else.\n")

    add("## Per-fold results (full pipeline)\n")
    add(fold_table(folds, "full"))
    add("")
    add(f"**{n_positive} of {n_folds} folds** were positive net of costs, with fold "
        f"Sharpe ranging {sharpe_spread}. Dispersion this wide across folds is the "
        f"reason a single test window is not evidence.\n")

    add("## Ablation: what do the two agents contribute?\n")
    add("The paper's Tables 7 and 8 report that removing either agent degrades "
        "performance, with the confidence agent mattering more.\n")
    add("| Configuration | Cum. return (net) | Sharpe (net) | vs full |")
    add("|---|---|---|---|")
    for key in ("full", "no_csa", "no_rpa", "no_agents_all_alphas"):
        if key not in set(summary["label"]):
            continue
        value = stat(key, "cum_return")
        delta = "-" if key == "full" else f"{(value - full_net) * 100:+.2f} pp"
        add(f"| {LABELS[key]} | {_pct(value)} | {_num(stat(key, 'sharpe'))} | {delta} |")
    add("")

    add("## Benchmarks and simpler baselines\n")
    add("Beating the index in a rising market is a weak claim. The question is "
        "whether the pipeline beats the things you would try first, on the same "
        "universe, portfolio rule and costs.\n")
    add("| Strategy | Cum. return (net) | Sharpe (net) |")
    add("|---|---|---|")
    for key in ("full", "single_alpha", "gbm", "no_agents_all_alphas",
                "universe_equal_weight", "sp500_equal_weight", "sp500_cap_weight"):
        if key in set(summary["label"]):
            add(f"| {LABELS[key]} | {_pct(stat(key, 'cum_return'))} | {_num(stat(key, 'sharpe'))} |")
    add("")

    add("## Statistical significance\n")
    add("Point estimates from one backtest are not evidence. HAC (Newey-West) "
        "standard errors correct for the autocorrelation in daily strategy returns; "
        "the confidence interval comes from a block bootstrap that resamples "
        "21-day blocks, preserving that autocorrelation.\n")
    add(significance_table(sig))
    add("")
    add(f"For the full pipeline the mean daily return carries a HAC t-statistic of "
        f"**{_num(hac_t)}** (p = {_num(hac_p, 3)}), and the bootstrap 95% interval for "
        f"its Sharpe is **[{_num(ci_low)}, {_num(ci_high)}]**. Against an equal-weight "
        f"hold of the same universe, the excess return t-statistic is **{_num(ex_t)}** "
        f"(p = {_num(ex_p, 3)}).\n")

    add("## Is it alpha, or a sector and beta tilt?\n")
    add("A concentrated 13-stock book can look like alpha while simply being long "
        "beta or long one sector. Strategy returns are regressed on the market and "
        "then on the market plus equal-weight GICS sector portfolios, with HAC "
        "errors. The intercept is what the boring explanations do not account for.\n")
    add(attribution_table(attrib))
    add("")
    add(f"The full pipeline's market beta is **{_num(beta)}**. Its CAPM alpha is "
        f"**{_pct(capm_alpha)}** a year (t = {_num(capm_t)}); adding sector factors "
        f"moves that to **{_pct(sector_alpha)}** (t = {_num(sector_t)}).\n")

    add("## Sensitivity to the agent weights (paper's Table 9)\n")
    if not sweep.empty:
        add("The paper reports 0.6/0.4 as clearly best, with a strikingly "
            "non-monotonic ladder around it (Sharpe 8.10 at 1.0/0.0, -2.68 at 0.8/0.2, "
            "11.39 at 0.6/0.4, -5.10 at 0.5/0.5). Re-run on this universe:\n")
        add(sweep_table(sweep))
        add("")
        best = sweep.loc[sweep["sharpe"].idxmax()]
        add(f"Best ratio here is **{best['w_confidence']:.1f}/{best['w_risk']:.1f}** "
            f"(net Sharpe {_num(best['sharpe'])}), against the paper's 0.6/0.4. "
            f"The spread across the whole ladder is "
            f"{_num(sweep['sharpe'].min())} to {_num(sweep['sharpe'].max())} — "
            f"worth weighing against the possibility that picking the best cell of a "
            f"seven-row sweep is itself a selection effect.\n")
    else:
        add("_Sweep not run._\n")

    add("## Which alphas the agents actually picked\n")
    add(selection_frequency(selections, "full"))
    add("")

    (REPORTS / "RESULTS.md").write_text("\n".join(L))
    print(f"wrote {REPORTS / 'RESULTS.md'} ({len(L)} lines)")


if __name__ == "__main__":
    main()
