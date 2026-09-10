"""Registry and compute engine for the Seed Alpha Factory (Stage 1).

An alpha is an :class:`AlphaSpec`: a name, a category, the paper's own formula
string, and a function of an :class:`AlphaContext`. Keeping the formula string
next to the implementation makes the transcription auditable -- you can diff
what we compute against Appendix A.3 line by line.

The engine deliberately reports three kinds of problem rather than hiding them:

* **unavailable** -- the formula needs data this environment cannot source
  (quote-level bid/ask, most FRED series). Emitted as NaN and excluded.
* **degenerate** -- the factor is constant across the cross-section on
  essentially every day, so it cannot rank stocks and its cross-sectional IC is
  undefined. All ten A.3 macro factors are in this bucket by construction.
* **duplicate** -- two A.3 entries reduce to the same signal (ROC and Momentum
  Oscillator are literally the same expression; Bollinger Bands and Percent B
  differ by a factor of 100). Kept, but flagged, since a "category-diversified"
  selection that picks two identical factors is not diversified.

Normalisation is cross-sectional and same-day only, so it cannot leak
information across time. That matters: A.3's factors are wildly heteroscedastic
(``VOLUME * CLOSE`` is order 1e9, ``EPS / CLOSE`` is order 1e-2), and both the
MLP and any linear combination need them on a common scale.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from alpha_lab.alphas import operators as ops
from alpha_lab.config import Config
from alpha_lab.data.panel import MarketPanel

logger = logging.getLogger(__name__)


@dataclass
class AlphaContext:
    """Everything an alpha function is allowed to see.

    Exposing the panel through a narrow context (rather than passing the panel
    itself) keeps alpha definitions honest: there is no route from here to a
    future date, a forward return, or another stock's cross-section.
    """

    panel: MarketPanel
    cfg: Config

    def f(self, name: str) -> pd.DataFrame:
        return self.panel.get(name)

    # Price/volume shorthands so formulas read like the paper's notation.
    @property
    def open(self) -> pd.DataFrame:
        return self.panel["open"]

    @property
    def high(self) -> pd.DataFrame:
        return self.panel["high"]

    @property
    def low(self) -> pd.DataFrame:
        return self.panel["low"]

    @property
    def close(self) -> pd.DataFrame:
        return self.panel["close"]

    @property
    def volume(self) -> pd.DataFrame:
        return self.panel["volume"]

    @property
    def vwap(self) -> pd.DataFrame:
        return self.panel["vwap"]

    @property
    def returns(self) -> pd.DataFrame:
        return self.panel["returns"]

    def macro(self, name: str) -> pd.DataFrame:
        """Broadcast a macro series across every ticker.

        This is what makes the A.3 macro category cross-sectionally constant:
        the same number in every column. Implemented faithfully, then detected
        by :func:`find_degenerate` rather than quietly special-cased.
        """
        if name not in self.panel.macro.columns:
            return self.panel.empty()
        series = self.panel.macro[name].reindex(self.panel.dates)
        return pd.DataFrame(
            np.repeat(series.to_numpy()[:, None], len(self.panel.tickers), axis=1),
            index=self.panel.dates,
            columns=self.panel.tickers,
        )


@dataclass(frozen=True)
class AlphaSpec:
    name: str
    category: str
    formula: str
    """The Appendix A.3 short code, verbatim, for audit."""

    fn: Callable[[AlphaContext], pd.DataFrame]
    available: bool = True
    notes: str = ""
    duplicate_of: str | None = None
    scale_free: bool = True
    """False when the factor carries units (dollars, shares, dollar-volume).

    Unit-bearing factors are not comparable across stocks: ``CLOSE -
    DELAY(CLOSE, 14)`` is a dollar change, so a $500 stock dominates a $20 one
    for reasons that have nothing to do with expected return. A.3 is full of
    these, and it materially affects what the pipeline can learn."""


class AlphaRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, AlphaSpec] = {}

    def add(self, spec: AlphaSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate alpha name: {spec.name}")
        self._specs[spec.name] = spec

    def register(self, **kwargs) -> Callable:
        def decorator(fn: Callable[[AlphaContext], pd.DataFrame]):
            self.add(AlphaSpec(fn=fn, **kwargs))
            return fn

        return decorator

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __getitem__(self, name: str) -> AlphaSpec:
        return self._specs[name]

    @property
    def specs(self) -> list[AlphaSpec]:
        return list(self._specs.values())

    def by_category(self, category: str) -> list[AlphaSpec]:
        return [s for s in self._specs.values() if s.category == category]

    @property
    def categories(self) -> list[str]:
        seen: list[str] = []
        for s in self._specs.values():
            if s.category not in seen:
                seen.append(s.category)
        return seen

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "name": s.name,
                    "category": s.category,
                    "formula": s.formula,
                    "available": s.available,
                    "scale_free": s.scale_free,
                    "duplicate_of": s.duplicate_of,
                    "notes": s.notes,
                }
                for s in self._specs.values()
            ]
        )


@dataclass
class AlphaPanel:
    """Computed alphas plus the diagnostics needed to trust them."""

    raw: dict[str, pd.DataFrame]
    normalized: dict[str, pd.DataFrame]
    registry: AlphaRegistry
    diagnostics: pd.DataFrame
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return sorted(self.normalized)

    def usable(self) -> list[str]:
        """Alphas fit to feed the agent layer: available, non-degenerate, and
        with enough non-null cross-sectional coverage to rank anything."""
        d = self.diagnostics
        ok = d[(~d["degenerate"]) & (d["available"]) & (d["pct_valid"] >= 20.0)]
        return sorted(ok["name"].tolist())

    def stack(self, names: list[str], dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
        """Long-format (date, ticker) x alpha matrix for model fitting."""
        frames = []
        for name in names:
            df = self.normalized[name]
            if dates is not None:
                df = df.loc[df.index.isin(dates)]
            frames.append(df.stack(future_stack=True).rename(name))
        out = pd.concat(frames, axis=1)
        out.index.names = ["date", "ticker"]
        return out


def compute_alphas(
    registry: AlphaRegistry, panel: MarketPanel, cfg: Config
) -> AlphaPanel:
    """Evaluate every registered alpha over the panel, then normalise."""
    ctx = AlphaContext(panel=panel, cfg=cfg)
    raw: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}

    for spec in registry.specs:
        if not spec.available and not cfg.alphas.include_unavailable:
            continue
        try:
            values = spec.fn(ctx)
        except Exception as exc:  # a broken formula must not kill the run
            logger.warning("alpha %s failed: %s", spec.name, exc)
            failures[spec.name] = f"{type(exc).__name__}: {exc}"
            continue
        values = values.reindex(index=panel.dates, columns=panel.tickers)
        raw[spec.name] = values.replace([np.inf, -np.inf], np.nan).astype(float)

    # Alphas are computed on the full panel, then masked to tradeable names.
    # Doing it in this order stops a stock's index entry/exit from truncating
    # its own moving averages.
    mask = panel.tradeable
    normalized: dict[str, pd.DataFrame] = {}
    for name, values in raw.items():
        masked = values.where(mask)
        if cfg.alphas.winsorize_quantile > 0:
            masked = ops.cs_winsorize(masked, cfg.alphas.winsorize_quantile)
        if cfg.alphas.cross_sectional_normalize == "zscore":
            normalized[name] = ops.cs_zscore(masked, cfg.alphas.min_cross_section)
        elif cfg.alphas.cross_sectional_normalize == "rank":
            ranked = ops.cs_rank(masked)
            counts = masked.notna().sum(axis=1)
            normalized[name] = ranked.where(counts >= cfg.alphas.min_cross_section)
        else:
            normalized[name] = masked

    diagnostics = _diagnose(raw, normalized, registry, panel)
    return AlphaPanel(
        raw=raw, normalized=normalized, registry=registry,
        diagnostics=diagnostics, failures=failures,
    )


def _diagnose(
    raw: dict[str, pd.DataFrame],
    normalized: dict[str, pd.DataFrame],
    registry: AlphaRegistry,
    panel: MarketPanel,
) -> pd.DataFrame:
    """Per-alpha health check: coverage, degeneracy, and price-level contamination."""
    mask = panel.tradeable
    n_tradeable = mask.to_numpy().sum()
    close = panel["close"].where(mask)
    log_price = np.log(close.where(close > 0))

    rows = []
    for name, values in raw.items():
        spec = registry[name]
        masked = values.where(mask)
        valid = masked.notna().to_numpy().sum()
        pct_valid = 100.0 * valid / max(n_tradeable, 1)

        # A factor that is constant across the cross-section cannot rank stocks.
        cs_std = masked.std(axis=1)
        cs_count = masked.notna().sum(axis=1)
        live = cs_count >= 2
        degenerate = bool(live.sum() == 0 or (cs_std[live].fillna(0) <= 1e-12).mean() > 0.99)

        # How much of the factor is just "this stock has a high price"? A high
        # |rho| means the factor is a price-level proxy in disguise.
        norm = normalized.get(name)
        price_rho = np.nan
        if norm is not None and not degenerate:
            per_day = _daily_rank_corr(norm, log_price)
            if per_day.notna().any():
                price_rho = float(per_day.mean())

        rows.append(
            {
                "name": name,
                "category": spec.category,
                "available": spec.available,
                "scale_free": spec.scale_free,
                "duplicate_of": spec.duplicate_of,
                "pct_valid": pct_valid,
                "degenerate": degenerate,
                "mean_abs_value": float(np.nanmean(np.abs(masked.to_numpy()))) if valid else np.nan,
                "corr_with_log_price": price_rho,
            }
        )
    return pd.DataFrame(rows).sort_values(["category", "name"]).reset_index(drop=True)


def _daily_rank_corr(a: pd.DataFrame, b: pd.DataFrame, min_obs: int = 20) -> pd.Series:
    """Per-day Spearman correlation between two aligned panels."""
    ra = a.rank(axis=1)
    rb = b.rank(axis=1)
    both = ra.notna() & rb.notna()
    ra, rb = ra.where(both), rb.where(both)
    n = both.sum(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    cov = (ra * rb).sum(axis=1)
    denom = np.sqrt((ra**2).sum(axis=1) * (rb**2).sum(axis=1))
    out = cov / denom.where(denom > 0)
    return out.where(n >= min_obs)


def find_degenerate(alpha_panel: AlphaPanel) -> list[str]:
    d = alpha_panel.diagnostics
    return sorted(d.loc[d["degenerate"], "name"].tolist())
