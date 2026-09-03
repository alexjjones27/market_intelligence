"""Macro regime classification: a simple 2x2 growth x inflation grid.

Per the build spec: never label a release "good" or "bad" in isolation.
A hot CPI print means something different in a disinflating-growth regime
than in an overheating one. This module classifies the *prevailing*
regime from recent growth and inflation releases, then derives an
expected cross-asset direction from the regime -- not from any single
headline number.

MVP simplification: growth state is read off the most recent GDP /
nonfarm payrolls / retail sales trend (rising vs. falling vs. their own
trailing average); inflation state off CPI / core CPI / PCE trend. This
is intentionally coarse -- a real implementation would blend several
indicators with lags and revisions -- but it keeps the classifier cheap,
deterministic, and testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GrowthState = Literal["up", "down"]
InflationState = Literal["up", "down"]

REGIME_LABELS: dict[tuple[GrowthState, InflationState], str] = {
    ("up", "up"): "overheat",
    ("up", "down"): "reflation",
    ("down", "up"): "stagflation",
    ("down", "down"): "disinflationary_slowdown",
}

# Expected cross-asset direction by regime. Directional bias only --
# not a trading signal, just context for interpreting a given event's
# market-sensitivity score.
EXPECTED_CROSS_ASSET_DIRECTION: dict[str, dict[str, str]] = {
    "overheat": {
        "equities": "mixed_to_negative",
        "rates": "up",
        "usd": "up",
        "commodities": "up",
        "growth_vs_value": "value_favored",
    },
    "reflation": {
        "equities": "positive",
        "rates": "flat_to_up",
        "usd": "flat",
        "commodities": "up",
        "growth_vs_value": "cyclicals_favored",
    },
    "stagflation": {
        "equities": "negative",
        "rates": "up",
        "usd": "mixed",
        "commodities": "up",
        "growth_vs_value": "defensives_favored",
    },
    "disinflationary_slowdown": {
        "equities": "mixed_to_positive",
        "rates": "down",
        "usd": "down",
        "commodities": "down",
        "growth_vs_value": "growth_favored",
    },
}


@dataclass(frozen=True)
class RegimeAssessment:
    growth_state: GrowthState
    inflation_state: InflationState
    regime_label: str
    expected_cross_asset_direction: dict[str, str]


def classify_trend(latest: float | None, prior: float | None) -> str | None:
    """up if latest > prior, down if latest < prior, None if we can't tell."""
    if latest is None or prior is None:
        return None
    if latest > prior:
        return "up"
    if latest < prior:
        return "down"
    return "down"  # flat treated as non-accelerating -> "down" side of the grid


def classify_regime(
    growth_latest: float | None,
    growth_prior: float | None,
    inflation_latest: float | None,
    inflation_prior: float | None,
) -> RegimeAssessment | None:
    growth_state = classify_trend(growth_latest, growth_prior)
    inflation_state = classify_trend(inflation_latest, inflation_prior)
    if growth_state is None or inflation_state is None:
        return None

    label = REGIME_LABELS[(growth_state, inflation_state)]  # type: ignore[index]
    return RegimeAssessment(
        growth_state=growth_state,  # type: ignore[arg-type]
        inflation_state=inflation_state,  # type: ignore[arg-type]
        regime_label=label,
        expected_cross_asset_direction=EXPECTED_CROSS_ASSET_DIRECTION[label],
    )


def classify_regime_from_releases(releases: dict[str, dict]) -> RegimeAssessment | None:
    """`releases` maps release_name -> {'value': float, 'prior_period_value': float}.

    Growth proxy: NONFARM_PAYROLLS trend (falls back to RETAIL_SALES, then GDP).
    Inflation proxy: CORE_CPI trend (falls back to CPI, then CORE_PCE).
    """
    growth_release = (
        releases.get("NONFARM_PAYROLLS") or releases.get("RETAIL_SALES") or releases.get("GDP")
    )
    inflation_release = (
        releases.get("CORE_CPI") or releases.get("CPI") or releases.get("CORE_PCE")
    )
    if not growth_release or not inflation_release:
        return None

    return classify_regime(
        growth_release.get("value"),
        growth_release.get("prior_period_value"),
        inflation_release.get("value"),
        inflation_release.get("prior_period_value"),
    )
