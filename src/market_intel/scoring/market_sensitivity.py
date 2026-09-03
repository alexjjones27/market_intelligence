"""Market sensitivity scoring: how exposed is this asset to the
macro/sector variable(s) the event touches.

Per the build spec, MVP starts with what's cheaply computable --
historical beta and static sector membership (see
exposure/mapping.py) -- and leaves the rest as clearly-marked
extensible stubs: revenue exposure by geography, rate duration,
currency exposure, index weight / passive-flow relevance. Each needs a
data source not available for free (segment reporting extraction,
duration analytics, index provider data) and is deliberately not
faked with a plausible-looking number.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field


def returns_from_prices(prices: list[float]) -> list[float]:
    return [
        (prices[i] - prices[i - 1]) / prices[i - 1]
        for i in range(1, len(prices))
        if prices[i - 1] not in (0, None) and prices[i] is not None
    ]


def compute_beta(asset_returns: list[float], benchmark_returns: list[float], min_points: int = 10) -> float | None:
    """OLS beta of asset returns on benchmark returns: cov(a,b) / var(b)."""
    n = min(len(asset_returns), len(benchmark_returns))
    if n < min_points:
        return None
    a, b = asset_returns[-n:], benchmark_returns[-n:]
    bench_var = statistics.variance(b)
    if bench_var == 0:
        return None
    cov = statistics.covariance(a, b)
    return round(cov / bench_var, 4)


RATE_SENSITIVITY_COMPONENT = {"high": 1.0, "medium": 0.5, "low": 0.15}


@dataclass
class MarketSensitivityResult:
    score: float               # 0..1
    label: str                 # low | medium | high
    beta: float | None
    components: dict = field(default_factory=dict)


def composite_market_sensitivity(
    beta: float | None,
    sector_rate_sensitivity: str | None,
    commodities: list[str],
    fx_sensitivity: list[str],
) -> MarketSensitivityResult:
    beta_component = min(abs(beta) / 2.0, 1.0) if beta is not None else 0.5
    rate_component = RATE_SENSITIVITY_COMPONENT.get(sector_rate_sensitivity or "", 0.5)
    commodity_component = 1.0 if commodities else 0.0
    fx_component = min(len(fx_sensitivity) / 2.0, 1.0)

    components = {
        "beta": beta_component,
        "rate_sensitivity": rate_component,
        "commodity_exposure": commodity_component,
        "fx_exposure": fx_component,
    }
    score = round(sum(components.values()) / len(components), 4)
    label = "high" if score >= 0.66 else "medium" if score >= 0.33 else "low"
    return MarketSensitivityResult(score=score, label=label, beta=beta, components=components)


# --- Extensible stubs -------------------------------------------------
# Each of these is a real, named input from the build spec that has no
# free/cheap data source for the MVP. They return None rather than a
# fabricated number, and are wired for a future real implementation.

def revenue_exposure_by_geography_stub(ticker: str) -> dict | None:
    """STUB: needs segment-reporting data parsed from 10-K geographic
    disclosures (Item 7 / segment notes). Not implemented in MVP."""
    return None


def rate_duration_stub(ticker: str) -> float | None:
    """STUB: needs balance-sheet duration analytics (debt maturity
    schedule, fixed vs floating mix). Not implemented in MVP."""
    return None


def index_passive_flow_relevance_stub(ticker: str) -> float | None:
    """STUB: needs index membership + weight data from an index
    provider (S&P, MSCI, Russell) plus ETF creation/redemption flow
    data. Not implemented in MVP."""
    return None
