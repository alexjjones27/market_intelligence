"""Surprise scoring.

Per-metric raw surprise:
    surprise = (actual - consensus) / max(abs(consensus), epsilon)

Standardized against each company's own trailing-N-quarter surprise
distribution (z-score) so a 5% surprise is comparable across companies
with very different typical surprise magnitudes (a company that
routinely beats by 15% shouldn't get the same score as one that never
misses by more than 1%).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

EPSILON = 1e-6

# guidance_direction -> a directional surprise proxy. Withdrawal is
# treated as maximally negative regardless of prior direction: it's
# always a loss of information/confidence.
GUIDANCE_DIRECTION_SCORE = {
    "raised": 1.0,
    "maintained": 0.0,
    "lowered": -1.0,
    "withdrawn": -1.0,
}


def raw_surprise(actual: float | None, consensus: float | None, epsilon: float = EPSILON) -> float | None:
    if actual is None or consensus is None:
        return None
    return (actual - consensus) / max(abs(consensus), epsilon)


def zscore(value: float | None, history: list[float], min_history: int = 2) -> float | None:
    """Standardize `value` against a trailing history of the same metric
    for the same company. Returns None if there isn't enough history to
    standardize against (falls back to raw surprise upstream)."""
    if value is None or len([h for h in history if h is not None]) < min_history:
        return None
    clean_history = [h for h in history if h is not None]
    mean = statistics.mean(clean_history)
    stdev = statistics.pstdev(clean_history)
    if stdev == 0:
        return 0.0
    return (value - mean) / stdev


def guidance_direction_score(direction: str | None) -> float | None:
    if direction is None:
        return None
    return GUIDANCE_DIRECTION_SCORE.get(direction)


def _squash(z: float, scale: float = 2.5) -> float:
    """Map an unbounded z-score-ish value into (-1, 1) via tanh, so the
    composite score stays interpretable regardless of how extreme a
    single metric's z-score gets."""
    import math

    return math.tanh(z / scale)


@dataclass
class MetricSurprise:
    metric: str
    raw: float | None
    standardized: float | None  # z-score if history available, else None


DEFAULT_WEIGHTS = {"eps": 0.45, "revenue": 0.30, "guidance": 0.25}


def composite_surprise(
    eps_z: float | None,
    revenue_z: float | None,
    guidance_score: float | None,
    weights: dict[str, float] = None,
) -> float | None:
    """Weighted blend of standardized per-metric surprises into one
    score in (-1, 1). Missing metrics are dropped and remaining weights
    renormalized. Returns None only if every input is None."""
    weights = weights or DEFAULT_WEIGHTS
    parts: dict[str, float] = {}
    if eps_z is not None:
        parts["eps"] = _squash(eps_z)
    if revenue_z is not None:
        parts["revenue"] = _squash(revenue_z)
    if guidance_score is not None:
        parts["guidance"] = guidance_score  # already in [-1, 1]

    if not parts:
        return None

    total_weight = sum(weights[k] for k in parts)
    blended = sum(weights[k] * v for k, v in parts.items()) / total_weight
    return round(blended, 4)


def surprise_score_0_to_1(composite: float | None) -> float | None:
    """Rescale the (-1, 1) composite surprise into a (0, 1) magnitude
    score for use in alert-threshold gating, where direction is tracked
    separately (earnings_direction)."""
    if composite is None:
        return None
    return round(abs(composite), 4)


# --- Per-component direction & coverage ---------------------------------
#
# The blended composite above is for RANKING. It must never be the sole
# basis for a directional label: an EPS miss alongside a revenue beat is
# not "negative," it's mixed, and forcing a binary label there is
# actively misleading. These functions classify direction from
# per-component signs directly.

DIRECTION_DEADBAND = 0.005  # +/-0.5% raw surprise is noise, not signal


def direction_from_value(value: float | None, deadband: float = DIRECTION_DEADBAND) -> str:
    """positive | negative | neutral | unknown, from a raw signed surprise."""
    if value is None:
        return "unknown"
    if value > deadband:
        return "positive"
    if value < -deadband:
        return "negative"
    return "neutral"


GUIDANCE_DIRECTION_LABEL = {"raised": "positive", "maintained": "neutral", "lowered": "negative", "withdrawn": "negative"}


def guidance_direction_label(direction: str | None) -> str:
    if direction is None:
        return "unknown"
    return GUIDANCE_DIRECTION_LABEL.get(direction, "unknown")


def combine_component_directions(directions: list[str]) -> str:
    """positive | negative | mixed | inconclusive | unknown, from a list
    of per-component 'positive'|'negative'|'neutral'|'unknown' directions.
    Two components with opposite signs always resolve to 'mixed' -- never
    forced into a single positive/negative label."""
    known = [d for d in directions if d != "unknown"]
    if not known:
        return "unknown"
    signed = {d for d in known if d in ("positive", "negative")}
    if len(signed) == 2:
        return "mixed"
    if signed == {"positive"}:
        return "positive"
    if signed == {"negative"}:
        return "negative"
    return "inconclusive"  # everything known fell inside the deadband, or too little signal


@dataclass
class SurpriseAssessment:
    composite: float | None    # blended, z-scored where possible -- for ranking only
    magnitude: float | None    # abs(composite) -- for threshold gating
    eps_surprise: float | None      # raw, signed
    revenue_surprise: float | None  # raw, signed
    direction: str              # positive | negative | mixed | inconclusive | unknown
    coverage: float             # fraction of {eps, revenue, guidance} with data available
    components_missing: list[str]


def score_surprise(
    eps_actual: float | None,
    eps_consensus: float | None,
    revenue_actual: float | None,
    revenue_consensus: float | None,
    guidance_direction: str | None,
    eps_history: list[float] | None = None,
    revenue_history: list[float] | None = None,
) -> SurpriseAssessment:
    eps_raw = raw_surprise(eps_actual, eps_consensus)
    revenue_raw = raw_surprise(revenue_actual, revenue_consensus)

    eps_z = zscore(eps_raw, eps_history or [])
    revenue_z = zscore(revenue_raw, revenue_history or [])
    guidance_score = guidance_direction_score(guidance_direction)

    composite = composite_surprise(eps_z, revenue_z, guidance_score)
    if composite is None:
        # Not enough trailing history to z-score yet -- fall back to raw
        # surprise so the score isn't thrown away, just not yet
        # comparable across companies.
        composite = composite_surprise(eps_raw, revenue_raw, guidance_score)

    direction = combine_component_directions(
        [
            direction_from_value(eps_raw),
            direction_from_value(revenue_raw),
            guidance_direction_label(guidance_direction),
        ]
    )

    missing = []
    if eps_raw is None:
        missing.append("eps")
    if revenue_raw is None:
        missing.append("revenue")
    if guidance_direction is None:
        missing.append("guidance")
    coverage = round((3 - len(missing)) / 3, 4)

    return SurpriseAssessment(
        composite=composite,
        magnitude=surprise_score_0_to_1(composite),
        eps_surprise=eps_raw,
        revenue_surprise=revenue_raw,
        direction=direction,
        coverage=coverage,
        components_missing=missing,
    )
