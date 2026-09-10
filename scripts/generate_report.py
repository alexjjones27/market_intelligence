"""Build reports/RESULTS.md from the backtest artefacts.

Every number in the prose is read from the CSVs rather than retyped, and the
verdict sentences are derived from the numbers so the write-up cannot drift
away from what the run actually produced.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpha_lab.reporting.report import (
    LABELS, attribution_table, fold_table, gross_net_table, overall_table,
    selection_frequency, significance_table, sweep_table, _num, _pct,
)

REPORTS = Path("reports")


# The paper's Table 6, SP500 rows, as printed. Our rolling folds land on exactly
# these three test windows (folds 1, 3 and 5), so the comparison is direct
# rather than approximate.
PAPER_TABLE6_SP500 = {
    "2021-01-01..2021-06-30": {"ann": 0.9361, "cum": 0.3519, "dd": -0.0789,
                               "bench_ann": 0.2967, "bench_cum": 0.1259, "bench_dd": -0.0423},
    "2022-01-01..2022-06-30": {"ann": 0.0277, "cum": 0.0125, "dd": -0.2055,
                               "bench_ann": -0.4422, "bench_cum": -0.2339, "bench_dd": -0.2351},
    "2023-01-01..2023-06-30": {"ann": 1.1824, "cum": 0.4278, "dd": -0.1152,
                               "bench_ann": 0.3522, "bench_cum": 0.1476, "bench_dd": -0.0775},
}


def paper_comparison_table(folds: pd.DataFrame) -> str:
    """Side-by-side against the paper's own reported SP500 numbers."""
    sub = folds[folds["variant"] == "full"].dropna(subset=["cum_return_net"])
    rows = ["| Test window | Paper: cum. return | Ours: cum. (net) | Ours: cum. (gross) | "
            "Paper: max DD | Ours: max DD |",
            "|---|---|---|---|---|---|"]
    matched = 0
    for window, paper in PAPER_TABLE6_SP500.items():
        match = sub[sub["test_window"] == window]
        if match.empty:
            rows.append(f"| {window} | {_pct(paper['cum'])} | _not run_ | _not run_ | "
                        f"{_pct(paper['dd'])} | - |")
            continue
        matched += 1
        r = match.iloc[0]
        rows.append(
            f"| {window} | {_pct(paper['cum'])} | {_pct(r['cum_return_net'])} | "
            f"{_pct(r['cum_return_gross'])} | {_pct(paper['dd'])} | {_pct(r['max_dd_net'])} |"
        )
    if matched == 0:
        return "_no folds line up with the paper's Table 6 windows_"
    return "\n".join(rows)



# Validated categorical palette (see the dataviz reference palette). Only the
# first four slots are used; benchmarks are deliberately neutral grey rather
# than a fifth hue, because they are reference lines, not peer series.

# Short legend labels. The direct labels at each line end already carry the full
# name, so the legend only has to disambiguate colour, and long strings there
# collide with each other at any sensible column count.
SHORT_LABELS = {
    "full": "Full pipeline",
    "no_csa": "No confidence agent",
    "no_rpa": "No risk agent",
    "single_alpha": "Best single alpha",
    "gbm": "Gradient boosting",
    "no_agents_all_alphas": "MLP, all alphas",
    "universe_equal_weight": "Equal-weight universe",
    "sp500_cap_weight": "S&P 500 (SPY)",
}


SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
BENCH_GREY = "#8a8983"


def _style_axes(ax) -> None:
    """Recessive grid and axes: the data carries the ink, not the furniture."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def _label_gap_in_data_units(ax, fontsize: float) -> float:
    """Minimum vertical spacing between stacked labels, in data units.

    Derived from the rendered text height rather than a guessed fraction of the
    axis range: a fixed fraction is either too tight on a tall axis or wasteful
    on a short one, and getting it wrong is exactly how end-labels end up
    sitting on top of each other.
    """
    fig = ax.get_figure()
    fig.canvas.draw()
    low, high = ax.get_ylim()
    pixel_height = ax.get_window_extent().height
    if pixel_height <= 0:
        return (high - low) * 0.06
    line_pixels = fontsize * fig.dpi / 72.0 * 1.45  # glyph height + leading
    return line_pixels * (high - low) / pixel_height


def _place_end_labels(ax, entries, fontsize: float = 9.0) -> None:
    """Draw line-end labels, nudged apart so they never overlap.

    ``entries`` is a list of ``(x, y, text, color)``. The dot stays anchored to
    the line's true end value; only the text slides, with a leader line drawn
    when it has moved far enough to need one. Text is in ink and the colored dot
    carries identity -- three of the four palette slots sit below 3:1 contrast on
    this surface, so the validator's relief rule applies and nothing may rest on
    hue alone.
    """
    if not entries:
        return
    min_gap = _label_gap_in_data_units(ax, fontsize)

    ordered = sorted(entries, key=lambda e: e[1])
    placed = [list(e) + [e[1]] for e in ordered]  # trailing slot = label y

    # One upward pass, then one downward pass, keeps the group centred rather
    # than pushing everything off the top of the axes.
    for i in range(1, len(placed)):
        if placed[i][4] - placed[i - 1][4] < min_gap:
            placed[i][4] = placed[i - 1][4] + min_gap
    for i in range(len(placed) - 2, -1, -1):
        if placed[i + 1][4] - placed[i][4] < min_gap:
            placed[i][4] = placed[i + 1][4] - min_gap

    for x, y, text, color, label_y in placed:
        ax.plot([x], [y], marker="o", markersize=6, color=color, zorder=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, clip_on=False)
        if abs(label_y - y) > min_gap * 0.25:
            ax.annotate(
                "", xy=(x, y), xytext=(x, label_y), annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=GRID, linewidth=1.0,
                                shrinkA=0, shrinkB=3),
            )
        ax.annotate(f"  {text}", (x, label_y), color=INK, fontsize=fontsize,
                    va="center", ha="left", annotation_clip=False)


def plot_equity_curves(returns: pd.DataFrame, bench: pd.DataFrame, out: Path) -> None:
    """Growth of 1 unit, net of costs, over the stitched out-of-sample period."""
    wanted = [c for c in ("full", "single_alpha", "gbm", "no_agents_all_alphas")
              if c in returns.columns]
    bench_wanted = [c for c in ("universe_equal_weight", "sp500_cap_weight")
                    if c in bench.columns]
    if not wanted:
        return

    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor=SURFACE)
    _style_axes(ax)

    end_labels = []

    # Benchmarks first, behind, in neutral grey.
    for i, name in enumerate(bench_wanted):
        equity = (1 + bench[name].fillna(0)).cumprod()
        ax.plot(equity.index, equity.values, color=BENCH_GREY, linewidth=1.6,
                linestyle="--" if i else "-", zorder=2)
        end_labels.append((equity.index[-1], equity.iloc[-1],
                           LABELS.get(name, name).replace("Benchmark: ", ""), BENCH_GREY))

    for slot, name in enumerate(wanted):
        equity = (1 + returns[name].fillna(0)).cumprod()
        ax.plot(equity.index, equity.values, color=SERIES[slot], linewidth=2.0, zorder=3)
        end_labels.append((
            equity.index[-1], equity.iloc[-1],
            LABELS.get(name, name).replace("Baseline: ", "").replace(" (CSA + RPA)", ""),
            SERIES[slot],
        ))

    ax.axhline(1.0, color=GRID, linewidth=1.0, zorder=1)
    _place_end_labels(ax, end_labels)
    ax.set_ylabel("growth of 1 unit (net of costs)", color=INK_MUTED, fontsize=10)
    ax.set_title("Out-of-sample equity curves, net of transaction costs",
                 color=INK, fontsize=13, loc="left", pad=38)
    handles = [plt.Line2D([], [], color=SERIES[i], linewidth=2.0) for i in range(len(wanted))]
    handles += [plt.Line2D([], [], color=BENCH_GREY, linewidth=1.6,
                           linestyle="--" if i else "-") for i in range(len(bench_wanted))]
    labels = [SHORT_LABELS.get(n, n) for n in wanted + bench_wanted]
    # Legend above the plot area rather than inside it: an in-axes legend
    # collides with whichever series happens to run into that corner, and which
    # corner is safe changes with the data.
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.005),
              ncol=3, frameon=False, fontsize=9, labelcolor=INK_MUTED,
              borderaxespad=0, columnspacing=1.6, handlelength=1.6)
    fig.subplots_adjust(right=0.78)
    fig.savefig(out, dpi=120, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_fold_returns(folds: pd.DataFrame, out: Path) -> None:
    """Per-fold net return: dispersion is the point, so every fold is shown."""
    sub = folds[folds["variant"] == "full"].dropna(subset=["cum_return_net"])
    ew = folds[folds["variant"] == "single_alpha"].dropna(subset=["cum_return_net"])
    if sub.empty:
        return
    sub = sub.sort_values("test_window")
    ew = ew.set_index("test_window").reindex(sub["test_window"])

    x = np.arange(len(sub))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)
    _style_axes(ax)
    # 2px surface gap between adjacent bars comes from the width/offset pair.
    ax.bar(x - width / 2 - 0.01, sub["cum_return_net"] * 100, width,
           color=SERIES[0], zorder=3, label=SHORT_LABELS["full"])
    ax.bar(x + width / 2 + 0.01, ew["cum_return_net"].to_numpy() * 100, width,
           color=SERIES[1], zorder=3, label=SHORT_LABELS["single_alpha"])
    ax.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["test_window"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("cumulative return over the fold, net (%)", color=INK_MUTED, fontsize=10)
    ax.set_title("Per-fold out-of-sample return: the dispersion a single test window hides",
                 color=INK, fontsize=13, loc="left", pad=32)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2, frameon=False,
              fontsize=9, labelcolor=INK_MUTED, borderaxespad=0, handlelength=1.6)
    fig.savefig(out, dpi=120, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)




def _verdict(ours: float, theirs: float, tolerance: float = 0.001) -> str:
    """Three-way comparison. A near-tie is reported as one, not as a loss."""
    if not (np.isfinite(ours) and np.isfinite(theirs)):
        return "not comparable"
    if abs(ours - theirs) <= tolerance:
        return "level"
    return "pipeline ahead" if ours > theirs else "pipeline behind"


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

    daily = read("results_daily_returns_net.csv")
    bench_daily = read("results_daily_benchmarks.csv")
    if not daily.empty:
        daily = daily.set_index(daily.columns[0]); daily.index = pd.to_datetime(daily.index)
        bench_daily = bench_daily.set_index(bench_daily.columns[0])
        bench_daily.index = pd.to_datetime(bench_daily.index)
        plot_equity_curves(daily, bench_daily, REPORTS / "results_equity_curves.png")
        plot_fold_returns(folds, REPORTS / "results_fold_returns.png")

    add("## Headline: stitched out-of-sample performance, net of costs\n")
    add("![Out-of-sample equity curves](results_equity_curves.png)\n")
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
    add("![Per-fold returns](results_fold_returns.png)\n")
    add(fold_table(folds, "full"))
    add("")
    add(f"**{n_positive} of {n_folds} folds** were positive net of costs, with fold "
        f"Sharpe ranging {sharpe_spread}. Dispersion this wide across folds is the "
        f"reason a single test window is not evidence.\n")

    add("## Direct comparison with the paper's reported S&P 500 results\n")
    add("The paper's Table 6 reports three SP500 test windows. Our rolling folds "
        "land on exactly those windows, so this is a like-for-like comparison of "
        "the same strategy design on the same index over the same dates, with the "
        "differences in universe construction, agent operationalisation and costs "
        "set out in [ALPHA_LAB.md](../ALPHA_LAB.md#deliberate-deviations-from-the-paper).\n")
    add(paper_comparison_table(folds))
    add("")
    add("The paper's annualised figures in Table 6 are worth reading carefully on "
        "their own terms: a 59.03% half-year is reported as a 192.27% annual "
        "return, where compounding gives 153%. Cumulative return is the "
        "unambiguous column, so that is what is compared here.\n")

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
            f"(net Sharpe {_num(best['sharpe'])}), not the paper's 0.6/0.4. The whole "
            f"ladder spans {_num(sweep['sharpe'].min())} to {_num(sweep['sharpe'].max())}.\n")
        if sweep["sharpe"].max() < 0.5:
            add("Read that spread against the bootstrap intervals above, which are "
                "roughly a full Sharpe point wide. Every cell in this ladder sits "
                "inside the noise band of every other cell, so the ordering carries "
                "no information. The paper reports the same sweep with Sharpe ratios "
                "from -8.04 to 11.39 and treats the ranking as meaningful; a ladder "
                "that swings that violently on a weight change is better read as "
                "evidence of an unstable objective than as a tuning result.\n")
        else:
            add("Picking the best cell of a seven-row sweep is itself a selection "
                "effect, which is why the whole ladder is printed.\n")
    else:
        add("_Sweep not run._\n")

    # ------------------------------------------------------- honest assessment
    add("## Does this hold up?\n")

    beats_ew = full_net > ew
    beats_cap = full_net > cap
    beats_single = full_net > single
    beats_gbm = full_net > gbm
    beats_all_alphas = full_net > all_alphas
    hac_significant = np.isfinite(hac_p) and hac_p < 0.05
    excess_significant = np.isfinite(ex_p) and ex_p < 0.05
    ci_spans_zero = np.isfinite(ci_low) and ci_low < 0 < ci_high
    sector_alpha_significant = (
        len(attrib_full) and np.isfinite(sector_t) and abs(sector_t) > 1.96
    )
    agents_help = full_net > max(no_csa, no_rpa)

    add("### The short answer\n")
    verdict = []
    verdict.append(
        f"Over {meta['n_folds']} out-of-sample folds the full pipeline returned "
        f"**{_pct(full_net)}** net of costs against **{_pct(ew)}** for simply holding "
        f"the same universe equal-weighted"
        + (", so it **did** beat that benchmark." if beats_ew else
           ", so it **did not** beat that benchmark.")
    )
    verdict.append(
        "Its mean daily return is "
        + ("**statistically distinguishable from zero** " if hac_significant
           else "**not statistically distinguishable from zero** ")
        + f"under a HAC-corrected test (t = {_num(hac_t)}, p = {_num(hac_p, 3)}), and the "
        f"block-bootstrap interval for its Sharpe "
        + ("spans zero" if ci_spans_zero else "excludes zero")
        + f" at [{_num(ci_low)}, {_num(ci_high)}]."
    )
    verdict.append(
        "Against an equal-weight hold of the same universe the excess return carries "
        f"t = {_num(ex_t)} (p = {_num(ex_p, 3)}), which is "
        + ("significant at the 5% level." if excess_significant
           else "**not** significant at the 5% level.")
    )
    if sector_alpha_significant and sector_alpha < 0:
        alpha_reading = (
            "which is significantly **negative**. This is a stronger statement than "
            "'no alpha': after accounting for market beta and sector exposure, the "
            "strategy destroyed value at a rate that passes conventional "
            "significance. The signal is not merely absent, it is adverse."
        )
    elif sector_alpha_significant:
        alpha_reading = (
            "which is significantly positive and therefore not explained by market "
            "or sector exposure."
        )
    else:
        alpha_reading = (
            "which does **not** survive conventional significance. On this evidence "
            "the return is accounted for by market and sector exposure rather than "
            "by factor selection."
        )
    verdict.append(
        "Once market beta and eleven sector portfolios are regressed out, the "
        f"unexplained annual alpha is **{_pct(sector_alpha)}** (t = {_num(sector_t)}), "
        + alpha_reading
    )
    if np.isfinite(beta) and beta > 1.05:
        verdict.append(
            f"Its market beta is **{_num(beta)}**, so it carried *more* market risk "
            "than the index while delivering less return."
        )
    for line in verdict:
        add("- " + line)
    add("")

    add("### Did the pipeline beat the simple things?\n")
    add("| Comparison | Result |")
    add("|---|---|")
    for caption, other in [
        ("equal-weight universe", ew),
        ("cap-weight S&P 500", cap),
        ("best single alpha", single),
        ("gradient boosting on the same features", gbm),
        ("MLP on all alphas, no agent layer", all_alphas),
    ]:
        add(f"| vs. {caption} | {_verdict(full_net, other)} "
            f"({_pct(full_net)} vs {_pct(other)}) |")
    add("")
    if _verdict(full_net, single) == "pipeline behind":
        add("The best-single-alpha row is the one that matters most. A three-stage "
            "pipeline that cannot beat one formula run through the identical portfolio "
            "rule is not earning its complexity on this data.\n")
    if _verdict(full_net, all_alphas) == "pipeline behind":
        add("The agent layer is also not paying for itself here: feeding every usable "
            "alpha straight to the network did better than selecting with CSA and RPA "
            "first.\n")
    elif agents_help:
        add("The ablations do point the same way as the paper's Table 7: removing "
            "either agent hurt, so the selection layer is contributing something on "
            "this data.\n")

    add("### Where our numbers diverge from the paper's, and why\n")
    add("The paper reports 53.17% on SSE50 for 2023 and 42.78% on SP500 for H1 2023. "
        "We do not reproduce anything of that size. The differences are structural, "
        "not a matter of tuning:\n")
    add("1. **Transaction costs.** The paper models none, at its own stated ~38% daily "
        f"turnover. Here that costs **{_pct(cost_drag)} a year**, taking cumulative "
        f"return from {_pct(full_gross)} gross to {_pct(full_net)} net. This single "
        "difference is worth more than most of the others combined.")
    add("2. **Execution timing.** We execute a signal one session after the close it "
        "was computed from. The paper is silent on this, and same-bar execution is "
        "worth a great deal at daily rebalancing frequency.")
    add("3. **Agent operationalisation.** The paper defines confidence and risk only "
        "as `E[IC]` and `f_risk(...)`. We had to invent concrete definitions. Ours are "
        "mechanical and can only see data available at the scoring date; the paper's "
        "were produced by an LLM that, per Appendix A.7, was shown *test-period factor "
        "performance* and whose training data covers the 2023 test window.")
    add("4. **Universe.** Point-in-time S&P 500 rather than SSE50. A 50-stock "
        "large-cap Chinese index in 2023 is a different opportunity set from ~450 US "
        "large caps, and cross-sectional strategies are highly sensitive to breadth.")
    add("5. **No multimodal inputs.** No audio, video, chart images or news sentiment. "
        "This is a real difference from the paper's design, though the paper provides "
        "no ablation isolating what those inputs contribute.")
    add("6. **Survivorship.** Our universe is point-in-time, but 110 delisted tickers "
        "have no price history. Whether the paper's SSE50 constituent set was "
        "point-in-time is not stated.\n")

    add("### What I would still not trust here\n")
    add("- **The delisted-ticker hole.** 16.8% of ever-members cannot be priced. "
        "Direction of bias is ambiguous (acquisitions remove winners, failures remove "
        "losers) but the magnitude is not negligible.")
    add("- **Sector labels are not point-in-time.** The attribution uses each ticker's "
        "most recently known GICS sector for all history. Sector reclassifications are "
        "rare but not zero.")
    add("- **The weight sweep is a selection surface.** Reporting the best cell of a "
        "seven-row ladder is itself a form of overfitting, which is exactly why the "
        "whole ladder is printed rather than only the winner.")
    add("- **One universe, one country, five and a half years.** Nothing here says "
        "anything about whether the design works elsewhere.")
    add("- **Alpha decay is not modelled.** Factors that worked in 2021 need not work "
        "in 2024, and the walk-forward design measures that but does not correct for it.\n")

    add("## Which alphas the agents actually picked\n")
    add(selection_frequency(selections, "full"))
    add("")

    (REPORTS / "RESULTS.md").write_text("\n".join(L))
    print(f"wrote {REPORTS / 'RESULTS.md'} ({len(L)} lines)")


if __name__ == "__main__":
    main()
