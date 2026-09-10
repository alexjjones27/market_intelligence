"""Full walk-forward backtest: pipeline, ablations, baselines, significance.

Writes reports/results_*.csv and the artefacts the results report is built from.
Everything is driven by configs/default.yaml; nothing here hard-codes a knob.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("alpha_lab.agents.selection").setLevel(logging.WARNING)

from alpha_lab.agents.factor_stats import compute_factor_stats
from alpha_lab.alphas.library import registry
from alpha_lab.alphas.registry import compute_alphas
from alpha_lab.backtest import baselines
from alpha_lab.backtest.engine import (
    VARIANTS, run_walk_forward, stitch_costs, stitch_returns, stitch_turnover,
)
from alpha_lab.backtest.metrics import summarise
from alpha_lab.backtest.statistics import (
    block_bootstrap, factor_attribution, hac_excess_test, hac_mean_test, sector_portfolios,
)
from alpha_lab.backtest.walkforward import generate_folds, paper_table5_folds
from alpha_lab.config import load_config
from alpha_lab.backtest.metrics import sharpe_ratio

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)
SCRATCH = Path("data/cache")


def load_inputs(cfg):
    """Panel, alphas and factor stats.

    Deliberately recomputed each run rather than cached: the normalised alpha
    panels are ~600MB in memory and the whole computation takes about two
    minutes, so caching would trade a lot of disk for very little time -- and a
    stale alpha cache silently invalidating a backtest is exactly the kind of
    bug this project exists to avoid.
    """
    panel = pickle.load(open(SCRATCH / "panel.pkl", "rb"))

    t0 = time.time()
    alphas = compute_alphas(registry, panel, cfg)
    logging.info("alphas computed in %.0fs (%d usable)", time.time() - t0, len(alphas.usable()))

    t0 = time.time()
    stats = compute_factor_stats(alphas, panel, cfg)
    logging.info("factor stats computed in %.0fs", time.time() - t0)
    return panel, alphas, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    cfg = load_config(args.config)
    panel, alphas, stats = load_inputs(cfg)

    folds = generate_folds(cfg, panel.dates.max())
    logging.info("%d walk-forward folds", len(folds))
    pd.DataFrame([f.describe() for f in folds]).to_csv(REPORTS / "results_folds.csv", index=False)

    t0 = time.time()
    runs = run_walk_forward(
        alphas, stats, panel, cfg, folds,
        variants=list(VARIANTS), include_baselines=True,
    )
    logging.info("walk-forward complete in %.0f min", (time.time() - t0) / 60)

    # ------------------------------------------------------------ per-fold table
    rows = []
    for run in runs:
        if run.result is None:
            rows.append({"fold": run.fold.name, "variant": run.variant,
                         "test_window": run.fold.label(), "note": run.notes})
            continue
        gross = summarise(run.result.gross_returns, label="gross",
                          risk_free=cfg.evaluation.risk_free_rate,
                          trading_days=cfg.evaluation.trading_days)
        net = summarise(run.result.net_returns, label="net",
                        turnover=run.result.turnover, costs=run.result.costs,
                        risk_free=cfg.evaluation.risk_free_rate,
                        trading_days=cfg.evaluation.trading_days)
        rows.append({
            "fold": run.fold.name,
            "variant": run.variant,
            "test_window": run.fold.label(),
            "n_selected": run.extras.get("n_selected"),
            "selected": ",".join(run.extras.get("selected", [])) or run.extras.get("alpha", ""),
            "cum_return_gross": gross["cum_return"],
            "cum_return_net": net["cum_return"],
            "ann_return_net": net["ann_return"],
            "ann_vol_net": net["ann_volatility"],
            "sharpe_gross": gross["sharpe"],
            "sharpe_net": net["sharpe"],
            "sortino_net": net["sortino"],
            "calmar_net": net["calmar"],
            "max_dd_net": net["max_drawdown"],
            "mean_daily_turnover": net["mean_daily_turnover"],
            "ann_cost_drag": net["ann_cost_drag"],
            "note": run.notes,
        })
    fold_table = pd.DataFrame(rows)
    fold_table.to_csv(REPORTS / "results_by_fold.csv", index=False)

    # ------------------------------------------------ stitched out-of-sample track
    variants = list(VARIANTS) + ["single_alpha", "gbm", "no_agents_all_alphas"]
    stitched_net = {v: stitch_returns(runs, v, net=True) for v in variants}
    stitched_gross = {v: stitch_returns(runs, v, net=False) for v in variants}
    stitched_turnover = {v: stitch_turnover(runs, v) for v in variants}
    stitched_costs = {v: stitch_costs(runs, v) for v in variants}

    oos_index = stitched_net["full"].index
    bench = baselines.benchmark_returns(panel).reindex(oos_index)
    bench["universe_equal_weight"] = baselines.equal_weight_universe_returns(panel).reindex(oos_index)

    summary_rows = []
    for name, series in stitched_net.items():
        if series.empty:
            continue
        s = summarise(series, label=name,
                      turnover=stitched_turnover.get(name),
                      costs=stitched_costs.get(name),
                      risk_free=cfg.evaluation.risk_free_rate,
                      trading_days=cfg.evaluation.trading_days)
        g = summarise(stitched_gross[name], label=name + "_gross",
                      risk_free=cfg.evaluation.risk_free_rate,
                      trading_days=cfg.evaluation.trading_days)
        s["cum_return_gross"] = g["cum_return"]
        s["sharpe_gross"] = g["sharpe"]
        s["kind"] = "strategy"
        summary_rows.append(s)
    for name in bench.columns:
        s = summarise(bench[name].dropna(), label=name,
                      risk_free=cfg.evaluation.risk_free_rate,
                      trading_days=cfg.evaluation.trading_days)
        s["kind"] = "benchmark"
        summary_rows.append(s)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORTS / "results_overall.csv", index=False)

    # --------------------------------------------------------------- significance
    sig_rows = []
    for name, series in stitched_net.items():
        if series.empty:
            continue
        hac = hac_mean_test(series)
        boot = block_bootstrap(
            series, lambda s: sharpe_ratio(s, cfg.evaluation.risk_free_rate,
                                           cfg.evaluation.trading_days),
            iterations=cfg.evaluation.bootstrap_iterations,
            block_size=cfg.evaluation.bootstrap_block_size,
            seed=cfg.evaluation.seed,
        )
        row = {
            "strategy": name,
            "mean_daily_return": hac.mean,
            "hac_t_stat": hac.t_stat,
            "hac_p_value": hac.p_value,
            "hac_lags": hac.lags,
            "sharpe_point": boot["point"],
            "sharpe_ci_low": boot["ci_low"],
            "sharpe_ci_high": boot["ci_high"],
            "bootstrap_p_sharpe_le_0": boot["p_value_gt_zero"],
        }
        for bname in ("sp500_cap_weight", "sp500_equal_weight", "universe_equal_weight"):
            if bname in bench.columns:
                ex = hac_excess_test(series, bench[bname])
                row[f"excess_vs_{bname}_daily"] = ex.mean
                row[f"excess_vs_{bname}_t"] = ex.t_stat
                row[f"excess_vs_{bname}_p"] = ex.p_value
        sig_rows.append(row)
    pd.DataFrame(sig_rows).to_csv(REPORTS / "results_significance.csv", index=False)

    # ------------------------------------------------ sector / beta decomposition
    sectors = sector_portfolios(panel["returns"], panel.sectors, panel.tradeable).reindex(oos_index)
    attrib_rows = []
    market = bench.get("sp500_cap_weight")
    for name, series in stitched_net.items():
        if series.empty or market is None:
            continue
        market_only = factor_attribution(series, market, None, cfg.evaluation.trading_days)
        with_sectors = factor_attribution(series, market, sectors, cfg.evaluation.trading_days)
        attrib_rows.append({
            "strategy": name,
            "capm_alpha_annual": market_only.alpha_annual,
            "capm_alpha_t": market_only.alpha_t_stat,
            "capm_alpha_p": market_only.alpha_p_value,
            "capm_beta": market_only.betas.get("market", np.nan),
            "capm_r2": market_only.r_squared,
            "sector_alpha_annual": with_sectors.alpha_annual,
            "sector_alpha_t": with_sectors.alpha_t_stat,
            "sector_alpha_p": with_sectors.alpha_p_value,
            "sector_model_r2": with_sectors.r_squared,
            "sector_beta_market": with_sectors.betas.get("market", np.nan),
        })
    pd.DataFrame(attrib_rows).to_csv(REPORTS / "results_attribution.csv", index=False)

    # ------------------------------------------------------------ selection audit
    sel_rows = []
    for run in runs:
        if run.selection is None or run.variant not in VARIANTS:
            continue
        for name in run.selection.alphas:
            r = run.selection.table.loc[name]
            sel_rows.append({
                "fold": run.fold.name, "variant": run.variant, "alpha": name,
                "category": r["category"], "mean_ic": r["mean_ic"],
                "confidence_pct": r["confidence_pct"], "risk_pct": r["risk_pct"],
                "score": r["score"],
            })
    pd.DataFrame(sel_rows).to_csv(REPORTS / "results_selections.csv", index=False)

    # ------------------------------------------------------ Table 9 weight sweep
    if not args.skip_sweep:
        sweep_rows = []
        ladder = [(1.0, 0.0), (0.8, 0.2), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.2, 0.8), (0.0, 1.0)]
        # One restart per fit here: the ladder is 7x the work of the main run and
        # the comparison across ratios is what matters, not each cell's last decimal.
        sweep_cfg_base = cfg.replace(mlp={"n_restarts": 1})
        for wc, wr in ladder:
            overrides = {"w_confidence": wc, "w_risk": wr,
                         "use_csa": wc > 0, "use_rpa": wr > 0}
            sweep_cfg = sweep_cfg_base.replace(selection=overrides)
            sweep_runs = run_walk_forward(
                alphas, stats, panel, sweep_cfg, folds, variants=["full"], include_baselines=False
            )
            series = stitch_returns(sweep_runs, "full", net=True)
            if series.empty:
                continue
            s = summarise(series, label=f"wc{wc}_wr{wr}",
                          risk_free=cfg.evaluation.risk_free_rate,
                          trading_days=cfg.evaluation.trading_days)
            s.update({"w_confidence": wc, "w_risk": wr})
            sweep_rows.append(s)
            logging.info("sweep wc=%.1f wr=%.1f -> net Sharpe %.3f", wc, wr, s["sharpe"])
        pd.DataFrame(sweep_rows).to_csv(REPORTS / "results_weight_sweep.csv", index=False)

    # ------------------------------------------------------ persist return series
    pd.DataFrame(stitched_net).to_csv(REPORTS / "results_daily_returns_net.csv")
    bench.to_csv(REPORTS / "results_daily_benchmarks.csv")

    meta = {
        "config": args.config,
        "n_folds": len(folds),
        "oos_start": str(oos_index.min().date()) if len(oos_index) else None,
        "oos_end": str(oos_index.max().date()) if len(oos_index) else None,
        "n_alphas_usable": len(alphas.usable()),
        "cost_bps_per_side": cfg.portfolio.cost_bps + cfg.portfolio.slippage_bps,
        "top_k": cfg.portfolio.top_k, "drop_n": cfg.portfolio.drop_n,
        "ic_horizon": cfg.csa.ic_horizon,
        "w_confidence": cfg.selection.w_confidence, "w_risk": cfg.selection.w_risk,
    }
    (REPORTS / "results_meta.json").write_text(json.dumps(meta, indent=2))

    print("\n=== OVERALL (stitched out-of-sample, net of costs) ===")
    cols = ["label", "kind", "cum_return", "ann_return", "ann_volatility", "sharpe",
            "sortino", "calmar", "max_drawdown"]
    print(summary[cols].round(4).to_string(index=False))
    print(f"\nwrote {REPORTS}/results_*.csv")


if __name__ == "__main__":
    main()
