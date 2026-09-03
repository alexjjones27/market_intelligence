from __future__ import annotations

from market_intel.scoring.market_confirmation import (
    abnormal_return,
    compute_confirmation_score,
    persistence_ratio,
    simple_return,
    volume_zscore,
)
from market_intel.scoring.market_sensitivity import compute_beta, composite_market_sensitivity, returns_from_prices
from market_intel.scoring.novelty import compute_novelty_score, guidance_shift_novelty, text_novelty


def test_simple_return_and_abnormal_return():
    r = simple_return(100.0, 105.0)
    assert r == 0.05
    assert abnormal_return(0.05, 0.01) == 0.04


def test_volume_zscore_flags_unusual_volume():
    history = [1_000_000] * 10
    z_normal = volume_zscore(1_000_000, history)
    z_spike = volume_zscore(4_000_000, history)
    assert z_normal == 0.0  # zero variance history -> 0
    history_varied = [900_000, 1_000_000, 1_100_000, 950_000, 1_050_000, 1_000_000]
    z = volume_zscore(3_000_000, history_varied)
    assert z is not None and z > 2


def test_persistence_ratio_reversal_vs_growth():
    growing = persistence_ratio(0.02, 0.05)
    reversal = persistence_ratio(0.02, -0.01)
    assert growing > 1.0
    assert reversal < 0


def test_confirmation_coverage_and_eligibility():
    full = compute_confirmation_score(abnormal_return_1d=0.03, volume_z=1.5, iv_change=0.1, persistence_1d_over_5m=1.0)
    assert full.coverage == 1.0
    assert full.components_missing == []
    assert full.is_eligible_for_alert is True

    thin = compute_confirmation_score(abnormal_return_1d=None, volume_z=None, iv_change=0.1, persistence_1d_over_5m=None)
    assert thin.coverage == 0.25
    assert set(thin.components_missing) == {"return", "volume", "persistence"}
    assert thin.is_eligible_for_alert is False

    empty = compute_confirmation_score(abnormal_return_1d=None, volume_z=None, iv_change=None, persistence_1d_over_5m=None)
    assert empty.score == 0.0
    assert empty.coverage == 0.0
    assert empty.is_eligible_for_alert is False


def test_confirmation_score_rewards_broad_persistent_move():
    loud_no_confirmation = compute_confirmation_score(
        abnormal_return_1d=0.001, volume_z=0.1, iv_change=0.0, persistence_1d_over_5m=0.1
    )
    quiet_confirmed = compute_confirmation_score(
        abnormal_return_1d=0.06, volume_z=3.5, iv_change=0.35, persistence_1d_over_5m=1.2
    )
    assert quiet_confirmed.score > loud_no_confirmation.score


def test_beta_computation_from_prices():
    prices_asset = [100, 102, 101, 105, 103, 108, 110, 107, 112, 115, 113]
    prices_bench = [100, 101, 100.5, 102, 101.5, 103, 104, 103, 105, 106, 105.5]
    r_a = returns_from_prices(prices_asset)
    r_b = returns_from_prices(prices_bench)
    beta = compute_beta(r_a, r_b)
    assert beta is not None
    assert beta > 0  # asset moves with benchmark, amplified


def test_beta_none_with_insufficient_data():
    assert compute_beta([0.01, 0.02], [0.01, 0.02]) is None


def test_composite_market_sensitivity_high_when_all_signals_high():
    result = composite_market_sensitivity(
        beta=1.8, sector_rate_sensitivity="high", commodities=["copper"], fx_sensitivity=["USD/TWD", "USD/JPY"]
    )
    assert result.label == "high"
    assert result.score > 0.66


def test_composite_market_sensitivity_low_when_all_signals_low():
    result = composite_market_sensitivity(
        beta=0.1, sector_rate_sensitivity="low", commodities=[], fx_sensitivity=[]
    )
    assert result.label == "low"


def test_novelty_first_disclosure_is_maximally_novel():
    assert compute_novelty_score(is_first_disclosure_of_type=True) == 1.0


def test_novelty_guidance_withdrawal_after_raise_is_highly_novel():
    score = guidance_shift_novelty("raised", "raised")  # same direction as before -> low
    withdrawal_score = guidance_shift_novelty("withdrawn", "raised")  # sudden withdrawal -> high
    assert withdrawal_score > score


def test_text_novelty_low_similarity_is_more_novel():
    prior = ["Example Corp reports strong quarterly earnings beat"]
    similar = text_novelty("Example Corp reports strong quarterly earnings beat", prior)
    different = text_novelty("Example Corp announces surprise CEO resignation", prior)
    assert different > similar


def test_novelty_score_blends_available_signals():
    score = compute_novelty_score(
        is_first_disclosure_of_type=False,
        guidance_direction="withdrawn",
        most_recent_prior_guidance_direction="raised",
        current_text="Example Corp withdraws full-year guidance amid demand uncertainty",
        prior_texts=["Example Corp raises full-year guidance on strong demand"],
    )
    assert score > 0.5
