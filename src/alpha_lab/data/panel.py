"""Assemble the aligned market panel every downstream stage reads from.

One object, one calendar, one column order. Every field -- prices, volumes,
fundamentals, derived valuation inputs -- is a ``DataFrame`` indexed by trading
date with tickers as columns, all reindexed to identical axes. Downstream code
can then do arithmetic between any two fields without worrying whether an
alignment silently dropped a name.

The panel also carries the two masks that decide what is *tradeable* on a given
day, kept separate on purpose:

* ``membership``  -- was the stock in the index that day (point-in-time)
* ``liquid``      -- did it clear the price and dollar-volume floors
* ``tradeable``   -- both, plus a valid price

Alphas are computed on the full panel and masked afterwards. Doing it the other
way round would let a name's exit from the universe truncate its own moving
averages, which changes the factor value for reasons unrelated to the market.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha_lab.config import Config
from alpha_lab.data.fundamentals import FundamentalData, add_derived_fields, load_fundamentals
from alpha_lab.data.macro import load_benchmarks, load_macro
from alpha_lab.data.prices import (
    PriceData, apply_liquidity_filter, load_prices, load_split_factors,
)
from alpha_lab.data.universe import UniverseData, load_universe, membership_coverage_report

logger = logging.getLogger(__name__)


@dataclass
class MarketPanel:
    fields: dict[str, pd.DataFrame]
    macro: pd.DataFrame
    benchmarks: pd.DataFrame
    sectors: pd.Series
    membership: pd.DataFrame
    liquid: pd.DataFrame
    tradeable: pd.DataFrame
    returns: pd.DataFrame
    diagnostics: dict[str, object] = field(default_factory=dict)

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.fields[name]

    def has(self, name: str) -> bool:
        return name in self.fields

    def get(self, name: str) -> pd.DataFrame:
        """Field lookup that yields an all-NaN frame for absent fields.

        Appendix A.3 references quantities we cannot source (bid/ask quotes,
        most FRED series). Returning NaN rather than raising lets the alpha
        library stay a literal transcription of the paper's table, with the
        gaps reported by the registry instead of hidden by omission."""
        if name in self.fields:
            return self.fields[name]
        return self.empty()

    def empty(self) -> pd.DataFrame:
        return pd.DataFrame(np.nan, index=self.dates, columns=self.tickers)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.fields["close"].index

    @property
    def tickers(self) -> list[str]:
        return list(self.fields["close"].columns)


def build_panel(cfg: Config, refresh: bool = False) -> MarketPanel:
    """Fetch everything and align it. This is the only place that touches IO."""
    universe = load_universe(cfg, refresh=refresh)
    tickers = universe.tickers
    logger.info("universe: %d distinct tickers across %d snapshots", len(tickers), len(universe.snapshot_dates))

    prices = load_prices(cfg, tickers, refresh=refresh)
    priced = set(prices.tickers)
    coverage = membership_coverage_report(universe, priced)
    missing_pct = coverage["pct_missing"].mean()
    if missing_pct > 1.0:
        logger.warning(
            "%.1f%% of point-in-time index members have no price history "
            "(typically delisted names Yahoo no longer serves). This reintroduces "
            "survivorship bias; see the coverage report in the results.",
            missing_pct,
        )

    dates = prices.dates
    columns = sorted(priced)

    def align(df: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(index=dates, columns=columns)

    fields: dict[str, pd.DataFrame] = {k: align(v) for k, v in prices.frames.items()}
    fields["vwap"] = _approx_vwap(fields)
    fields["returns"] = align(prices.returns)

    # Fetched before fundamentals because per-share XBRL figures have to be
    # restated onto the same split basis as the price series.
    split_factors = load_split_factors(cfg, columns, dates, refresh=refresh)
    fields["split_factor"] = align(split_factors)

    fundamentals: FundamentalData | None = None
    if cfg.data.fundamentals_source == "sec_edgar":
        ciks = universe.ciks.reindex(columns).dropna()
        fundamentals = load_fundamentals(
            cfg, ciks, dates, refresh=refresh, split_factors=split_factors
        )
        fundamentals = add_derived_fields(fundamentals, fields["close"])
        for name, frame in fundamentals.frames.items():
            fields[name] = align(frame)

    macro = load_macro(cfg, dates, refresh=refresh)
    benchmarks = load_benchmarks(cfg, dates, refresh=refresh)

    membership = universe.daily_mask(dates).reindex(columns=columns).fillna(False)
    liquid = apply_liquidity_filter(prices, cfg).reindex(index=dates, columns=columns).fillna(False)
    tradeable = membership & liquid & fields["close"].notna()

    backtest_start = pd.Timestamp(cfg.data.backtest_start)
    tradeable.loc[tradeable.index < backtest_start] = False

    diagnostics = {
        "universe_coverage": coverage,
        "price_failures": prices.failed,
        "fundamental_coverage": fundamentals.coverage if fundamentals else pd.DataFrame(),
        "fundamental_missing": fundamentals.missing if fundamentals else [],
        "n_tickers": len(columns),
        "n_dates": len(dates),
        "mean_tradeable_per_day": float(tradeable.loc[tradeable.index >= backtest_start].sum(axis=1).mean()),
    }
    logger.info(
        "panel: %d dates x %d tickers, %.0f tradeable names/day",
        len(dates), len(columns), diagnostics["mean_tradeable_per_day"],
    )

    return MarketPanel(
        fields=fields,
        macro=macro,
        benchmarks=benchmarks,
        sectors=universe.sectors.reindex(columns),
        membership=membership,
        liquid=liquid,
        tradeable=tradeable,
        returns=fields["returns"],
        diagnostics=diagnostics,
    )


def _approx_vwap(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Typical price as a VWAP stand-in.

    The paper lists VWAP among its six primary features. Daily bars do not carry
    a true volume-weighted average price, and Yahoo does not provide one, so we
    use (H+L+C)/3 -- the standard daily proxy. It is genuinely an approximation:
    on a day with a big open gap and heavy early volume it can be materially off.
    """
    return (fields["high"] + fields["low"] + fields["close"]) / 3.0
