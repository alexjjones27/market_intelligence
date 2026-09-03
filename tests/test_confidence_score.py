from __future__ import annotations

from market_intel.scoring.confidence import (
    ConfidenceInputs,
    compute_confidence,
    extraction_completeness,
    needs_verification,
)


def _base_inputs(**overrides) -> ConfidenceInputs:
    defaults = dict(
        primary_source_confirmed=True,
        num_corroborating_sources=4,
        extraction_completeness=1.0,
        timestamp_certainty=1.0,
        already_priced_in=0.0,
        opinion_fraction=0.0,
    )
    defaults.update(overrides)
    return ConfidenceInputs(**defaults)


def test_fully_confirmed_high_quality_event_scores_near_one():
    score = compute_confidence(_base_inputs())
    assert score >= 0.95


def test_unconfirmed_single_source_scores_much_lower():
    weak = _base_inputs(
        primary_source_confirmed=False,
        num_corroborating_sources=1,
        extraction_completeness=0.25,
        timestamp_certainty=0.3,
        opinion_fraction=0.8,
    )
    strong = _base_inputs()
    assert compute_confidence(weak) < compute_confidence(strong)


def test_already_priced_in_reduces_confidence():
    fresh = _base_inputs(already_priced_in=0.0)
    stale = _base_inputs(already_priced_in=1.0)
    assert compute_confidence(stale) < compute_confidence(fresh)


def test_opinion_heavy_content_reduces_confidence():
    factual = _base_inputs(opinion_fraction=0.0)
    opinion = _base_inputs(opinion_fraction=1.0)
    assert compute_confidence(opinion) < compute_confidence(factual)


def test_score_is_bounded_0_to_1():
    score = compute_confidence(_base_inputs())
    assert 0.0 <= score <= 1.0
    score_weak = compute_confidence(
        _base_inputs(
            primary_source_confirmed=False,
            num_corroborating_sources=0,
            extraction_completeness=0.0,
            timestamp_certainty=0.0,
            already_priced_in=1.0,
            opinion_fraction=1.0,
        )
    )
    assert 0.0 <= score_weak <= 1.0


def test_extraction_completeness_fraction():
    facts = {"eps_actual": 2.14, "eps_consensus": 1.92, "revenue_actual": None, "revenue_consensus": None}
    result = extraction_completeness(facts, ["eps_actual", "eps_consensus", "revenue_actual", "revenue_consensus"])
    assert result == 0.5


def test_extraction_completeness_no_expected_fields_is_neutral():
    assert extraction_completeness({}, []) == 0.5


def test_needs_verification_flags_high_impact_low_confidence():
    assert needs_verification(confidence=0.2, impact_score=0.9) is True


def test_needs_verification_false_for_high_confidence_high_impact():
    assert needs_verification(confidence=0.9, impact_score=0.9) is False


def test_needs_verification_false_for_low_impact_low_confidence():
    # low impact events don't need a verification flag even if confidence is thin --
    # they're just not going to trigger an alert in the first place.
    assert needs_verification(confidence=0.2, impact_score=0.1) is False


def test_needs_verification_true_when_inputs_missing():
    assert needs_verification(confidence=None, impact_score=0.9) is True
    assert needs_verification(confidence=0.9, impact_score=None) is True
