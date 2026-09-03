from __future__ import annotations

import math

from market_intel.scoring.surprise import (
    combine_component_directions,
    composite_surprise,
    direction_from_value,
    guidance_direction_label,
    guidance_direction_score,
    raw_surprise,
    score_surprise,
    surprise_score_0_to_1,
    zscore,
)


def test_raw_surprise_basic():
    # actual 2.14 vs consensus 1.92 -> beat by ~11.46%
    s = raw_surprise(2.14, 1.92)
    assert math.isclose(s, (2.14 - 1.92) / 1.92, rel_tol=1e-9)


def test_raw_surprise_handles_none():
    assert raw_surprise(None, 1.92) is None
    assert raw_surprise(2.14, None) is None


def test_raw_surprise_uses_epsilon_for_zero_consensus():
    s = raw_surprise(0.05, 0.0)
    assert s is not None
    assert s > 0


def test_zscore_standardizes_against_trailing_history():
    # company that historically beats by ~2% consistently; a 20% beat should be a big z
    history = [0.01, 0.02, 0.015, 0.025, 0.018]
    z_typical = zscore(0.02, history)
    z_huge = zscore(0.20, history)
    assert z_typical is not None and z_huge is not None
    assert z_huge > z_typical


def test_zscore_none_without_enough_history():
    assert zscore(0.05, [0.01]) is None
    assert zscore(0.05, []) is None


def test_zscore_zero_variance_history_returns_zero():
    assert zscore(0.05, [0.02, 0.02, 0.02]) == 0.0


def test_guidance_direction_score_mapping():
    assert guidance_direction_score("raised") == 1.0
    assert guidance_direction_score("lowered") == -1.0
    assert guidance_direction_score("withdrawn") == -1.0
    assert guidance_direction_score("maintained") == 0.0
    assert guidance_direction_score(None) is None


def test_composite_surprise_blends_available_metrics():
    composite = composite_surprise(eps_z=2.0, revenue_z=1.0, guidance_score=1.0)
    assert composite is not None
    assert -1.0 < composite < 1.0
    assert composite > 0  # positive surprise across the board -> positive composite


def test_composite_surprise_renormalizes_when_metric_missing():
    only_eps = composite_surprise(eps_z=2.0, revenue_z=None, guidance_score=None)
    with_all = composite_surprise(eps_z=2.0, revenue_z=2.0, guidance_score=1.0)
    assert only_eps is not None and with_all is not None


def test_composite_surprise_none_when_all_missing():
    assert composite_surprise(eps_z=None, revenue_z=None, guidance_score=None) is None


def test_surprise_score_0_to_1_is_a_magnitude():
    assert surprise_score_0_to_1(-0.5) == 0.5
    assert surprise_score_0_to_1(0.5) == 0.5
    assert surprise_score_0_to_1(None) is None


def test_bigger_beat_yields_higher_composite_than_smaller_beat():
    history = [0.0, 0.01, -0.01, 0.02, 0.0]
    small_beat_z = zscore(0.02, history)
    big_beat_z = zscore(0.30, history)
    small_composite = composite_surprise(eps_z=small_beat_z, revenue_z=None, guidance_score=None)
    big_composite = composite_surprise(eps_z=big_beat_z, revenue_z=None, guidance_score=None)
    assert big_composite > small_composite


# --- Per-component direction & mixed-result classification --------------


def test_direction_from_value_respects_deadband():
    assert direction_from_value(0.05) == "positive"
    assert direction_from_value(-0.05) == "negative"
    assert direction_from_value(0.001) == "neutral"
    assert direction_from_value(None) == "unknown"


def test_guidance_direction_label_mapping():
    assert guidance_direction_label("raised") == "positive"
    assert guidance_direction_label("lowered") == "negative"
    assert guidance_direction_label("withdrawn") == "negative"
    assert guidance_direction_label("maintained") == "neutral"
    assert guidance_direction_label(None) == "unknown"


def test_combine_component_directions_opposite_signs_is_mixed():
    assert combine_component_directions(["positive", "negative"]) == "mixed"
    assert combine_component_directions(["negative", "positive", "unknown"]) == "mixed"


def test_combine_component_directions_agreement():
    assert combine_component_directions(["positive", "positive", "unknown"]) == "positive"
    assert combine_component_directions(["negative", "negative"]) == "negative"


def test_combine_component_directions_all_unknown():
    assert combine_component_directions(["unknown", "unknown"]) == "unknown"


def test_combine_component_directions_all_neutral_is_inconclusive():
    assert combine_component_directions(["neutral", "neutral"]) == "inconclusive"
    assert combine_component_directions(["neutral", "unknown"]) == "inconclusive"


def test_score_surprise_eps_miss_revenue_beat_is_mixed_not_negative():
    # The exact NVDA-shaped case that motivated this: EPS 0.55 vs 0.63
    # consensus (a miss) alongside revenue 4.335B vs 4.25B consensus (a
    # beat) must NOT collapse into a single "negative" label.
    result = score_surprise(
        eps_actual=0.55, eps_consensus=0.63,
        revenue_actual=4_335_000_000, revenue_consensus=4_250_000_000,
        guidance_direction=None,
    )
    assert result.eps_surprise < 0
    assert result.revenue_surprise > 0
    assert result.direction == "mixed"
    assert result.components_missing == ["guidance"]
    assert round(result.coverage, 4) == round(2 / 3, 4)


def test_score_surprise_all_components_agree_positive():
    result = score_surprise(
        eps_actual=2.14, eps_consensus=1.92,
        revenue_actual=12_500_000_000, revenue_consensus=12_200_000_000,
        guidance_direction="raised",
    )
    assert result.direction == "positive"
    assert result.coverage == 1.0
    assert result.components_missing == []


def test_score_surprise_no_data_is_unknown():
    result = score_surprise(
        eps_actual=None, eps_consensus=None,
        revenue_actual=None, revenue_consensus=None,
        guidance_direction=None,
    )
    assert result.direction == "unknown"
    assert result.coverage == 0.0
    assert result.composite is None
    assert result.magnitude is None


def test_score_surprise_falls_back_to_raw_when_no_history():
    # No trailing history to z-score against -> composite still populated
    # from raw surprise rather than silently dropped.
    result = score_surprise(
        eps_actual=2.14, eps_consensus=1.92,
        revenue_actual=None, revenue_consensus=None,
        guidance_direction=None,
    )
    assert result.composite is not None
    assert result.magnitude is not None
