"""Leakage tests: the single most important guarantee in this repo.

The headline check is **truncation invariance**. Compute every alpha on the full
panel, then compute them again on a panel that has been cut off at some date T,
and compare the two on all dates up to T. If any operator looks forward -- a
negative shift, a centred rolling window, a normalisation that pools across
time, a fundamental keyed on fiscal period end rather than filing date -- the
values will differ, because the truncated run cannot see what the full run saw.

The last test in this module deliberately *introduces* a look-ahead alpha and
asserts the check catches it. A leakage test that cannot fail is worse than no
leakage test, because it manufactures confidence.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from alpha_lab.alphas.library import registry
from alpha_lab.alphas.registry import AlphaSpec, compute_alphas
from alpha_lab.data.panel import MarketPanel


def truncate(panel: MarketPanel, cutoff: pd.Timestamp) -> MarketPanel:
    """A panel that has never seen a single bar after ``cutoff``."""
    keep = panel.dates <= cutoff
    dates = panel.dates[keep]
    return dataclasses.replace(
        panel,
        fields={k: v.loc[dates] for k, v in panel.fields.items()},
        macro=panel.macro.loc[panel.macro.index <= cutoff],
        benchmarks=panel.benchmarks.loc[panel.benchmarks.index <= cutoff]
        if len(panel.benchmarks)
        else panel.benchmarks,
        membership=panel.membership.loc[dates],
        liquid=panel.liquid.loc[dates],
        tradeable=panel.tradeable.loc[dates],
        returns=panel.returns.loc[dates],
    )


def compare_truncation(full_df, cut_df, cutoff) -> tuple[float, int]:
    """Compare a full-panel alpha against its truncated-panel twin.

    Returns ``(max_abs_diff, n_vanished)`` where ``n_vanished`` counts cells the
    full run produced a value for but the truncated run could not.

    Both halves are necessary. Comparing only cells where both runs produced a
    value silently skips the boundary window -- and the boundary is precisely
    where a forward-looking operator shows itself, because the truncated run
    runs out of future data there while the full run does not. A backward-looking
    operator's value at date t depends only on data at or before t, all of which
    survives truncation, so it must produce the *same non-null pattern* too.
    """
    a = full_df.loc[:cutoff]
    b = cut_df.loc[:cutoff]
    a, b = a.align(b, join="inner")
    both = a.notna() & b.notna()
    vanished = int((a.notna() & b.isna()).to_numpy().sum())
    if not both.to_numpy().any():
        return 0.0, vanished
    diff = (a - b).abs().where(both).to_numpy()
    max_diff = float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0
    return max_diff, vanished


def test_alphas_are_truncation_invariant(panel, cfg):
    """No alpha value may change -- or disappear -- when future data is removed."""
    cutoff = panel.dates[300]
    full = compute_alphas(registry, panel, cfg)
    cut = compute_alphas(registry, truncate(panel, cutoff), cfg)

    offenders: dict[str, str] = {}
    for name in cut.normalized:
        max_diff, vanished = compare_truncation(
            full.normalized[name], cut.normalized[name], cutoff
        )
        if max_diff > 1e-8:
            offenders[name] = f"values differ by {max_diff:.3g}"
        elif vanished:
            offenders[name] = f"{vanished} cells needed future data"
    assert not offenders, f"look-ahead detected: {offenders}"


def test_truncation_check_actually_catches_a_leak(panel, cfg):
    """Meta-test: prove the invariance check above has teeth."""
    leaky = AlphaSpec(
        name="_deliberate_lookahead",
        category="momentum",
        formula="CLOSE shifted from the FUTURE (intentionally wrong)",
        fn=lambda c: c.close.shift(-5),
    )
    registry.add(leaky)
    try:
        cutoff = panel.dates[300]
        full = compute_alphas(registry, panel, cfg)
        cut = compute_alphas(registry, truncate(panel, cutoff), cfg)
        max_diff, vanished = compare_truncation(
            full.normalized["_deliberate_lookahead"],
            cut.normalized["_deliberate_lookahead"],
            cutoff,
        )
        assert max_diff > 1e-8 or vanished > 0, (
            "leakage check failed to detect an obvious look-ahead"
        )
    finally:
        registry._specs.pop("_deliberate_lookahead", None)


def test_no_alpha_uses_a_negative_shift():
    """Static guard: DELAY refuses negative n, but a raw .shift(-n) would slip
    past it. Nothing in the library should call shift with a negative literal."""
    import inspect

    from alpha_lab.alphas import library

    source = inspect.getsource(library)
    assert "shift(-" not in source


def test_rolling_windows_are_right_aligned():
    """pandas rolling is right-closed by default; assert it explicitly so a
    future refactor to center=True cannot pass silently."""
    from alpha_lab.alphas import operators as ops

    x = pd.DataFrame({"A": [1.0, 2, 3, 4, 5]}, index=pd.bdate_range("2020-01-01", periods=5))
    sma = ops.SMA(x, 3)["A"]
    # A centred window would put the mean of (1,2,3) at index 1, not index 2.
    assert np.isnan(sma.iloc[1])
    assert sma.iloc[2] == pytest.approx(2.0)


def test_cross_sectional_normalisation_does_not_pool_across_time(panel, cfg):
    """Rewriting future rows must not change any past normalised value."""
    computed = compute_alphas(registry, panel, cfg)
    cutoff_idx = 250

    scrambled = dataclasses.replace(
        panel,
        fields={
            k: pd.concat([v.iloc[:cutoff_idx], v.iloc[cutoff_idx:] * 1000.0])
            for k, v in panel.fields.items()
        },
    )
    scrambled_alphas = compute_alphas(registry, scrambled, cfg)

    name = "zscore_reversion"
    a = computed.normalized[name].iloc[:cutoff_idx]
    b = scrambled_alphas.normalized[name].iloc[:cutoff_idx]
    both = a.notna() & b.notna()
    assert float(np.nanmax((a - b).abs().where(both).to_numpy())) < 1e-8


def test_fundamentals_are_never_visible_before_their_filing_date():
    """A fact must enter the panel on its `filed` date, not its period end."""
    from alpha_lab.data.fundamentals import _facts_frame, _flow_known_series

    payload = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {
                    "units": {
                        "USD/shares": [
                            # Period ends in March; the market learns in May.
                            {"start": "2021-01-01", "end": "2021-03-31", "val": 1.0,
                             "filed": "2021-05-10", "form": "10-Q"},
                            {"start": "2021-04-01", "end": "2021-06-30", "val": 1.1,
                             "filed": "2021-08-09", "form": "10-Q"},
                            {"start": "2021-07-01", "end": "2021-09-30", "val": 1.2,
                             "filed": "2021-11-08", "form": "10-Q"},
                            {"start": "2021-10-01", "end": "2021-12-31", "val": 1.3,
                             "filed": "2022-02-14", "form": "10-K"},
                        ]
                    }
                }
            }
        }
    }
    series = _flow_known_series(_facts_frame(payload, "EarningsPerShareDiluted"))
    assert not series.empty
    # The full-year TTM is knowable only once the last quarter is filed.
    assert series.index.min() == pd.Timestamp("2022-02-14")
    assert series.iloc[0] == pytest.approx(4.6)
    # Nothing may be dated to a fiscal period end.
    assert pd.Timestamp("2021-12-31") not in series.index


def test_stale_fundamentals_expire_rather_than_persisting_forever():
    from alpha_lab.data.fundamentals import _to_trading_days

    known = pd.Series([5.0], index=[pd.Timestamp("2020-01-15")])
    days = pd.bdate_range("2020-01-01", "2021-12-31")
    out = _to_trading_days(known, days, max_staleness_days=400)
    assert out.loc["2020-06-01"] == pytest.approx(5.0)
    assert np.isnan(out.loc["2021-06-01"])


def test_alphas_are_masked_to_tradeable_names(panel, cfg):
    """A name that is not tradeable on a date must carry no normalised alpha."""
    blocked = panel.tickers[0]
    panel.tradeable[blocked] = False
    computed = compute_alphas(registry, panel, cfg)
    assert computed.normalized["rsi"][blocked].isna().all()
