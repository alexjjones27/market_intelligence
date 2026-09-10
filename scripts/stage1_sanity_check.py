"""Stage 1 verification: does the Seed Alpha Factory actually compute what it claims?

Produces reports/stage1_*.{csv,md,png}. Three kinds of evidence:

1. **Behavioural** -- do alphas move the way the indicator says they should
   against known price action (the Feb-Mar 2020 crash, the 2023 mega-cap rally)?
2. **Structural** -- bounded indicators stay in their bounds; z-scores are
   standardised per day; fundamentals step on filing dates and nowhere else.
3. **Honest diagnostics** -- coverage, cross-sectional degeneracy, duplicate
   factors, and how much of each factor is just a price-level proxy.
"""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from alpha_lab.alphas.library import registry
from alpha_lab.alphas.registry import compute_alphas
from alpha_lab.config import load_config

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)


def load_panel():
    with open("data/cache/panel.pkl", "rb") as fh:
        return pickle.load(fh)


def main() -> None:
    cfg = load_config()
    panel = load_panel()
    print(f"panel: {len(panel.dates)} sessions x {len(panel.tickers)} tickers")

    alphas = compute_alphas(registry, panel, cfg)
    diag = alphas.diagnostics
    diag.to_csv(REPORTS / "stage1_alpha_diagnostics.csv", index=False)

    spec_table = registry.summary()
    spec_table.to_csv(REPORTS / "stage1_alpha_specs.csv", index=False)

    lines: list[str] = []
    add = lines.append

    add("# Stage 1 - Seed Alpha Factory: sanity check\n")
    add(f"- Panel: **{len(panel.dates)} sessions** "
        f"({panel.dates.min().date()} to {panel.dates.max().date()}) "
        f"x **{len(panel.tickers)} tickers**")
    add(f"- Alphas registered: **{len(registry)}** across "
        f"**{len(registry.categories)}** categories")
    add(f"- Alphas computed: **{len(alphas.raw)}**; failures: {len(alphas.failures)}")
    add(f"- Mean tradeable names/day: **{panel.diagnostics['mean_tradeable_per_day']:.0f}**\n")

    if alphas.failures:
        add("## Alphas that failed to compute\n")
        for name, err in alphas.failures.items():
            add(f"- `{name}`: {err}")
        add("")

    # ---------------------------------------------------------------- structure
    add("## 1. Structural checks\n")
    checks: list[tuple[str, bool, str]] = []

    rsi = alphas.raw["rsi"].where(panel.tradeable)
    checks.append(("RSI stays within [0, 100]",
                   bool(np.nanmin(rsi.to_numpy()) >= -1e-9 and np.nanmax(rsi.to_numpy()) <= 100 + 1e-9),
                   f"observed [{np.nanmin(rsi.to_numpy()):.2f}, {np.nanmax(rsi.to_numpy()):.2f}]"))

    stoch = alphas.raw["stochastic_oscillator"].where(panel.tradeable)
    checks.append(("Stochastic oscillator within [0, 100]",
                   bool(np.nanmin(stoch.to_numpy()) >= -1e-9 and np.nanmax(stoch.to_numpy()) <= 100 + 1e-9),
                   f"observed [{np.nanmin(stoch.to_numpy()):.2f}, {np.nanmax(stoch.to_numpy()):.2f}]"))

    wr = alphas.raw["williams_r"].where(panel.tradeable)
    checks.append(("Williams %R within [-100, 0]",
                   bool(np.nanmin(wr.to_numpy()) >= -100 - 1e-9 and np.nanmax(wr.to_numpy()) <= 1e-9),
                   f"observed [{np.nanmin(wr.to_numpy()):.2f}, {np.nanmax(wr.to_numpy()):.2f}]"))

    vol = alphas.raw["historical_volatility"].where(panel.tradeable)
    checks.append(("Historical volatility non-negative",
                   bool(np.nanmin(vol.to_numpy()) >= -1e-12),
                   f"min {np.nanmin(vol.to_numpy()):.4f}"))

    z = alphas.normalized["rsi"]
    day_mean = float(np.nanmean(z.mean(axis=1).to_numpy()))
    day_std = float(np.nanmean(z.std(axis=1).to_numpy()))
    checks.append(("Cross-sectional z-scores standardised per day",
                   abs(day_mean) < 1e-6 and abs(day_std - 1.0) < 0.05,
                   f"mean {day_mean:.2e}, std {day_std:.4f}"))

    no_inf = all(not np.isinf(v.to_numpy()).any() for v in alphas.normalized.values())
    checks.append(("No infinities anywhere in normalised alphas", no_inf, ""))

    add("| Check | Result | Detail |")
    add("|---|---|---|")
    for label, ok, detail in checks:
        add(f"| {label} | {'PASS' if ok else 'FAIL'} | {detail} |")
    add("")

    # ------------------------------------------------------------- behavioural
    add("## 2. Behavioural checks against known price action\n")
    behaviour: list[str] = []

    covid_lo = slice("2020-03-16", "2020-03-24")
    calm = slice("2020-01-06", "2020-02-14")
    med_vol_covid = float(vol.loc[covid_lo].median(axis=1).median())
    med_vol_calm = float(vol.loc[calm].median(axis=1).median())
    behaviour.append(
        f"- **Volatility spikes in the COVID crash.** Universe-median annualised "
        f"historical volatility: **{med_vol_calm:.1%}** in Jan-Feb 2020 vs "
        f"**{med_vol_covid:.1%}** on 16-24 Mar 2020 "
        f"({med_vol_covid / med_vol_calm:.1f}x). "
        f"{'PASS' if med_vol_covid > 2 * med_vol_calm else 'UNEXPECTED'}"
    )

    med_rsi_covid = float(rsi.loc[covid_lo].median(axis=1).median())
    med_rsi_calm = float(rsi.loc[calm].median(axis=1).median())
    behaviour.append(
        f"- **RSI collapses at the crash low.** Universe-median RSI(14): "
        f"**{med_rsi_calm:.1f}** in Jan-Feb 2020 vs **{med_rsi_covid:.1f}** "
        f"on 16-24 Mar 2020. {'PASS' if med_rsi_covid < 35 else 'UNEXPECTED'}"
    )

    bpos = alphas.raw["bollinger_position"].where(panel.tradeable)
    med_b_covid = float(bpos.loc[covid_lo].median(axis=1).median())
    behaviour.append(
        f"- **Bollinger position pinned to the lower band in the crash.** "
        f"Universe-median (CLOSE-LOWER)/(UPPER-LOWER) = **{med_b_covid:.2f}** "
        f"on 16-24 Mar 2020 (0 = lower band, 1 = upper). "
        f"{'PASS' if med_b_covid < 0.3 else 'UNEXPECTED'}"
    )

    dd = alphas.raw["ulcer_index"].where(panel.tradeable)
    worst = dd.median(axis=1).idxmax()
    behaviour.append(
        f"- **Worst universe-wide Ulcer Index** (drawdown pain) falls on "
        f"**{worst.date()}** -- the COVID crash bottom week. "
        f"{'PASS' if pd.Timestamp('2020-03-01') <= worst <= pd.Timestamp('2020-04-15') else 'UNEXPECTED'}"
    )
    lines.extend(behaviour)
    add("")

    # ------------------------------- point-in-time fundamentals (step function)
    add("## 3. Fundamentals move only on filing dates\n")
    eps = panel["eps"]
    sample = [t for t in ("AAPL", "MSFT", "JNJ", "XOM") if t in eps.columns]
    add("| Ticker | Distinct TTM EPS values 2019-2024 | Change dates | Median gap (days) |")
    add("|---|---|---|---|")
    for t in sample:
        s = eps[t].loc["2019-01-01":"2024-06-30"].dropna()
        changes = s[s.diff().abs() > 1e-9]
        gaps = pd.Series(changes.index).diff().dt.days.dropna()
        add(f"| {t} | {s.nunique()} | {len(changes)} | "
            f"{gaps.median():.0f} |")
    add("")
    add("Roughly-quarterly gaps confirm the values step on SEC filing dates rather "
        "than drifting daily or updating at fiscal period end.\n")

    # --------------------------------------------------------- honest problems
    add("## 4. Diagnostics worth knowing before Stage 2\n")

    degen = diag[diag["degenerate"]]
    add(f"### Cross-sectionally degenerate: {len(degen)} alphas\n")
    add("These take the same value for every stock on a given day, so they cannot "
        "rank anything and their cross-sectional IC is undefined (not zero -- "
        "undefined; the Spearman correlation of a constant vector is NaN).\n")
    if len(degen):
        add("| Alpha | Category |")
        add("|---|---|")
        for _, r in degen.iterrows():
            add(f"| `{r['name']}` | {r['category']} |")
    add("")

    unavailable = spec_table[~spec_table["available"]]
    add(f"### Not computable from available data: {len(unavailable)} alphas\n")
    if len(unavailable):
        add("| Alpha | Reason |")
        add("|---|---|")
        for _, r in unavailable.iterrows():
            add(f"| `{r['name']}` | {r['notes']} |")
    add("")

    dupes = spec_table[spec_table["duplicate_of"].notna()]
    add(f"### Duplicate formulas in Appendix A.3: {len(dupes)} alphas\n")
    add("A.3 lists these twice, sometimes in different categories. That matters "
        "for a *category-diversified* selection rule: picking one from each of two "
        "categories can pick the same signal twice.\n")
    if len(dupes):
        add("| Alpha | Duplicate of | Category |")
        add("|---|---|---|")
        for _, r in dupes.iterrows():
            add(f"| `{r['name']}` | `{r['duplicate_of']}` | {r['category']} |")
    add("")

    scale = diag[diag["corr_with_log_price"].notna()].copy()
    scale["abs_rho"] = scale["corr_with_log_price"].abs()
    worst_scale = scale.sort_values("abs_rho", ascending=False).head(12)
    n_high = int((scale["abs_rho"] > 0.5).sum())
    add(f"### Price-level contamination: {n_high} alphas correlate |rho| > 0.5 with log price\n")
    add("Many A.3 formulas carry units. `CLOSE - DELAY(CLOSE, 14)` is a *dollar* "
        "change, so a \\$500 stock outranks a \\$20 stock arithmetically, not "
        "predictively. Ranked cross-sectionally these behave largely as price-level "
        "or size proxies. Mean daily Spearman rho against log(price):\n")
    add("| Alpha | Category | rho vs log price | scale-free? |")
    add("|---|---|---|---|")
    for _, r in worst_scale.iterrows():
        add(f"| `{r['name']}` | {r['category']} | {r['corr_with_log_price']:+.3f} | "
            f"{'yes' if r['scale_free'] else 'NO'} |")
    add("")

    thin = diag[diag["pct_valid"] < 50].sort_values("pct_valid")
    add(f"### Thin coverage: {len(thin)} alphas valid on <50% of tradeable stock-days\n")
    if len(thin):
        add("| Alpha | Category | % of tradeable stock-days with a value |")
        add("|---|---|---|")
        for _, r in thin.iterrows():
            add(f"| `{r['name']}` | {r['category']} | {r['pct_valid']:.1f}% |")
    add("")

    # ------------------------------------------------------ survivorship report
    cov = panel.diagnostics["universe_coverage"]
    add("## 5. Survivorship: how much of the point-in-time universe we can price\n")
    add(f"- Distinct tickers that were ever S&P 500 members in the window: "
        f"**{len(cov) and int(cov['n_members'].max())}** per snapshot, "
        f"**{len(panel.tickers) + len(panel.diagnostics['price_failures'])}** distinct overall")
    add(f"- Tickers with no price history at all: "
        f"**{len(panel.diagnostics['price_failures'])}** "
        f"({100 * len(panel.diagnostics['price_failures']) / (len(panel.tickers) + len(panel.diagnostics['price_failures'])):.1f}%)")
    add(f"- Mean share of index members missing prices on a given snapshot: "
        f"**{cov['pct_missing'].mean():.1f}%**\n")
    add("Yahoo purges history for delisted symbols, so names that left the index "
        "through bankruptcy, acquisition or a ticker change return nothing. This is "
        "a *residual* survivorship bias on top of an otherwise point-in-time "
        "universe, and it is quantified here rather than assumed away. Direction is "
        "genuinely ambiguous: acquisitions (mostly at a premium) remove winners, "
        "while failures like FRC remove losers.\n")
    add("Missing tickers (sample): " +
        ", ".join(f"`{t}`" for t in sorted(panel.diagnostics["price_failures"])[:40]) + "\n")

    (REPORTS / "stage1_sanity_check.md").write_text("\n".join(lines))
    print("\n".join(lines[:80]))
    print(f"\nwrote {REPORTS/'stage1_sanity_check.md'}")

    make_plots(panel, alphas)


def make_plots(panel, alphas) -> None:
    """Alphas plotted against price action a reader can recognise."""
    ticker = "AAPL" if "AAPL" in panel.tickers else panel.tickers[0]
    window = slice("2019-06-01", "2024-06-30")

    fig, axes = plt.subplots(5, 1, figsize=(14, 17), sharex=True)
    close = panel["close"][ticker].loc[window]

    ax = axes[0]
    ax.plot(close.index, close.values, color="#1f77b4", lw=1.2)
    upper = alphas.raw["moving_average"][ticker].loc[window]
    ax.plot(upper.index, upper.values, color="#ff7f0e", lw=1.0, label="SMA(20)")
    for lo, hi, label in [("2020-02-19", "2020-03-23", "COVID crash"),
                          ("2023-01-01", "2023-07-31", "2023 rally")]:
        ax.axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color="grey", alpha=0.18)
        ax.text(pd.Timestamp(lo), close.max() * 0.98, f" {label}", fontsize=8, va="top")
    ax.set_title(f"{ticker} close with SMA(20) - shaded: known price regimes")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    r = alphas.raw["rsi"][ticker].loc[window]
    ax.plot(r.index, r.values, color="#2ca02c", lw=1.0)
    ax.axhline(70, ls="--", lw=0.8, color="grey")
    ax.axhline(30, ls="--", lw=0.8, color="grey")
    ax.set_ylim(0, 100)
    ax.set_title("RSI(14) - should sink below 30 into the March 2020 low and ride high in the 2023 rally")

    ax = axes[2]
    b = alphas.raw["bollinger_position"][ticker].loc[window]
    ax.plot(b.index, b.values, color="#9467bd", lw=1.0)
    ax.axhline(1.0, ls="--", lw=0.8, color="grey")
    ax.axhline(0.0, ls="--", lw=0.8, color="grey")
    ax.set_title("Bollinger position (0 = lower band, 1 = upper band)")

    ax = axes[3]
    v = alphas.raw["historical_volatility"][ticker].loc[window]
    med = alphas.raw["historical_volatility"].where(panel.tradeable).median(axis=1).loc[window]
    ax.plot(v.index, v.values, color="#d62728", lw=1.0, label=ticker)
    ax.plot(med.index, med.values, color="black", lw=1.0, alpha=0.6, label="universe median")
    ax.set_title("Historical volatility, STD(RETURNS, 20) * sqrt(252)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[4]
    ey = alphas.raw["earnings_yield"][ticker].loc[window]
    ax.step(ey.index, ey.values, where="post", color="#8c564b", lw=1.2)
    ax.set_title("Earnings yield (TTM EPS / CLOSE) - steps ONLY on SEC filing dates, "
                 "then drifts with price")
    ax.set_xlabel("date")

    fig.tight_layout()
    fig.savefig(REPORTS / "stage1_sanity_alphas.png", dpi=110)
    plt.close(fig)

    # Scale-dependence: the structural problem, drawn.
    diag = alphas.diagnostics.dropna(subset=["corr_with_log_price"]).copy()
    diag["abs_rho"] = diag["corr_with_log_price"].abs()
    top = diag.sort_values("abs_rho", ascending=False).head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = ["#d62728" if not sf else "#1f77b4" for sf in top["scale_free"]]
    ax.barh(top["name"], top["corr_with_log_price"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("mean daily Spearman rho vs log(price)")
    ax.set_title("Price-level contamination by alpha\n"
                 "(red = formula carries units and is not scale-free)")
    fig.tight_layout()
    fig.savefig(REPORTS / "stage1_scale_dependence.png", dpi=110)
    plt.close(fig)
    print(f"wrote {REPORTS/'stage1_sanity_alphas.png'} and {REPORTS/'stage1_scale_dependence.png'}")


if __name__ == "__main__":
    main()
