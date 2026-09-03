"""Market-confirmation scoring: abnormal return vs. sector/market, volume
z-score, IV change, and persistence across 5m/1h/1d/5d horizons.

This is the module that separates a loud headline with no price
confirmation from a quiet one that produces a broad, persistent
repricing -- the whole point of pairing `facts`/`interpretation` with an
independently observed `market_reaction`.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field


def simple_return(price_before: float | None, price_after: float | None) -> float | None:
    if price_before in (None, 0) or price_after is None:
        return None
    return (price_after - price_before) / price_before


def abnormal_return(asset_return: float | None, benchmark_return: float | None) -> float | None:
    if asset_return is None or benchmark_return is None:
        return None
    return asset_return - benchmark_return


def volume_zscore(current_volume: float | None, historical_volumes: list[float], min_history: int = 5) -> float | None:
    clean = [v for v in historical_volumes if v is not None]
    if current_volume is None or len(clean) < min_history:
        return None
    mean = statistics.mean(clean)
    stdev = statistics.pstdev(clean)
    if stdev == 0:
        return 0.0
    return round((current_volume - mean) / stdev, 4)


def persistence_ratio(initial_return: float | None, later_return: float | None) -> float | None:
    """Fraction of the initial move retained at a later horizon.
    Same-direction and >= magnitude => ratio >= 1 (the move grew).
    Opposite sign => negative (a reversal/fade)."""
    if initial_return is None or later_return is None or initial_return == 0:
        return None
    return round(later_return / initial_return, 4)


@dataclass
class ConfirmationResult:
    score: float  # 0..1
    components: dict = field(default_factory=dict)


def compute_confirmation_score(
    abnormal_return_1d: float | None,
    volume_z: float | None,
    iv_change: float | None,
    persistence_1d_over_5m: float | None,
    sector_breadth: float | None = None,
) -> ConfirmationResult:
    components: dict[str, float] = {}
    if abnormal_return_1d is not None:
        components["return"] = min(abs(abnormal_return_1d) / 0.05, 1.0)  # >=5% abnormal move saturates
    if volume_z is not None:
        components["volume"] = min(max(volume_z, 0.0) / 3.0, 1.0)  # >=3 std saturates
    if iv_change is not None:
        components["iv"] = min(abs(iv_change) / 0.30, 1.0)  # >=30% IV change saturates
    if persistence_1d_over_5m is not None:
        components["persistence"] = min(max(persistence_1d_over_5m, 0.0), 1.0)
    if sector_breadth is not None:
        components["breadth"] = min(max(sector_breadth, 0.0), 1.0)

    if not components:
        return ConfirmationResult(score=0.0, components=components)
    score = round(sum(components.values()) / len(components), 4)
    return ConfirmationResult(score=score, components=components)
