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
    separately (forward_earnings_effect)."""
    if composite is None:
        return None
    return round(abs(composite), 4)
