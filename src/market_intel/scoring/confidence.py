"""Confidence scoring.

Combines: primary-source confirmation, corroborating-source count,
extraction certainty, timestamp certainty, whether the move looks
already priced in, and opinion-vs-fact content mix.

A high-impact (surprise/confirmation) event with LOW confidence must
produce a verification flag instead of an alert -- see
`needs_verification`. This is the guardrail against confidently-wrong
alerts from thin, single-source, poorly-timestamped signals.
"""
from __future__ import annotations

from dataclasses import dataclass

WEIGHTS = {
    "primary_source": 0.30,
    "corroboration": 0.20,
    "extraction": 0.20,
    "timestamp": 0.10,
    "not_priced_in": 0.10,
    "fact_based": 0.10,
}


@dataclass
class ConfidenceInputs:
    primary_source_confirmed: bool
    num_corroborating_sources: int          # total evidence count for the event
    extraction_completeness: float          # 0..1: fraction of expected fact fields populated
    timestamp_certainty: float              # 0..1: how precisely we know the actual event time
    # True = evidence of pre-event drift/anticipation; False = no such evidence found;
    # None = not assessed at all. None must NOT be treated as "not priced in" (0.0) --
    # that would silently claim evidence of absence where there's simply no evidence.
    # When None, this component is dropped entirely and the remaining weights
    # renormalize, rather than the score looking more precise than it is.
    already_priced_in: bool | None = None
    opinion_fraction: float = 0.0           # 0..1: higher = more opinion/analysis, less raw fact


def compute_confidence(inputs: ConfidenceInputs) -> float:
    primary_score = 1.0 if inputs.primary_source_confirmed else 0.3
    corroboration_score = min(1.0, inputs.num_corroborating_sources / 4)
    fact_score = 1.0 - _clamp(inputs.opinion_fraction)

    weighted: dict[str, tuple[float, float]] = {
        "primary_source": (WEIGHTS["primary_source"], primary_score),
        "corroboration": (WEIGHTS["corroboration"], corroboration_score),
        "extraction": (WEIGHTS["extraction"], _clamp(inputs.extraction_completeness)),
        "timestamp": (WEIGHTS["timestamp"], _clamp(inputs.timestamp_certainty)),
        "fact_based": (WEIGHTS["fact_based"], fact_score),
    }
    if inputs.already_priced_in is not None:
        weighted["not_priced_in"] = (WEIGHTS["not_priced_in"], 0.0 if inputs.already_priced_in else 1.0)

    total_weight = sum(w for w, _ in weighted.values())
    score = sum(w * v for w, v in weighted.values()) / total_weight
    return round(_clamp(score), 4)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def extraction_completeness(facts: dict, expected_fields: list[str]) -> float:
    if not expected_fields:
        return 0.5  # unknown event type with no defined expected fields -> neutral
    populated = sum(1 for f in expected_fields if facts.get(f) is not None)
    return round(populated / len(expected_fields), 4)


EXPECTED_FACT_FIELDS = {
    "earnings": ["eps_actual", "eps_consensus", "revenue_actual", "revenue_consensus"],
    "guidance_change": ["guidance_direction"],
}


def needs_verification(
    confidence: float | None,
    impact_score: float | None,
    confidence_threshold: float = 0.5,
    impact_threshold: float = 0.5,
) -> bool:
    """A high-impact, low-confidence event should raise a verification
    flag rather than fire an alert. `impact_score` is typically the
    surprise magnitude or a blend of surprise + confirmation."""
    if confidence is None or impact_score is None:
        return True  # can't assess -> err toward requiring a human look
    return impact_score >= impact_threshold and confidence < confidence_threshold
