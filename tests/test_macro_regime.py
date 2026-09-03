"""Explicit coverage for all four cells of the growth x inflation grid --
added after a review claimed a labeling bug ("Overheat" shown for a
growth-down/inflation-down state). That specific claim didn't reproduce
against the actual code or a live run (see git history / PR discussion),
but the grid deserved explicit per-quadrant tests regardless, since
nothing had pinned the mapping down before.
"""
from __future__ import annotations

from market_intel.scoring.macro_regime import (
    REGIME_LABELS,
    classify_regime,
    classify_regime_from_releases,
    classify_trend,
)


def test_growth_up_inflation_up_is_overheat():
    result = classify_regime(growth_latest=105, growth_prior=100, inflation_latest=105, inflation_prior=100)
    assert result.growth_state == "up"
    assert result.inflation_state == "up"
    assert result.regime_label == "overheat"


def test_growth_up_inflation_down_is_reflation():
    result = classify_regime(growth_latest=105, growth_prior=100, inflation_latest=95, inflation_prior=100)
    assert result.growth_state == "up"
    assert result.inflation_state == "down"
    assert result.regime_label == "reflation"


def test_growth_down_inflation_up_is_stagflation():
    result = classify_regime(growth_latest=95, growth_prior=100, inflation_latest=105, inflation_prior=100)
    assert result.growth_state == "down"
    assert result.inflation_state == "up"
    assert result.regime_label == "stagflation"


def test_growth_down_inflation_down_is_disinflationary_slowdown():
    result = classify_regime(growth_latest=95, growth_prior=100, inflation_latest=95, inflation_prior=100)
    assert result.growth_state == "down"
    assert result.inflation_state == "down"
    assert result.regime_label == "disinflationary_slowdown"


def test_regime_labels_table_covers_all_four_cells():
    assert set(REGIME_LABELS.keys()) == {("up", "up"), ("up", "down"), ("down", "up"), ("down", "down")}
    assert len(set(REGIME_LABELS.values())) == 4  # four distinct labels, no accidental collision


def test_classify_trend_flat_is_treated_as_down_not_up():
    # Documented modeling choice: a flat reading is "non-accelerating,"
    # grouped with "down" rather than invented as a third state.
    assert classify_trend(100.0, 100.0) == "down"


def test_classify_regime_none_without_both_series():
    assert classify_regime(None, 100, 100, 95) is None
    assert classify_regime(105, 100, None, 95) is None


def test_classify_regime_from_releases_uses_fallback_proxies():
    releases = {
        "RETAIL_SALES": {"value": 110, "prior_period_value": 100},
        "CPI": {"value": 95, "prior_period_value": 100},
    }
    result = classify_regime_from_releases(releases)
    assert result is not None
    assert result.regime_label == "reflation"


def test_classify_regime_from_releases_none_when_no_proxies_available():
    assert classify_regime_from_releases({}) is None
