"""Algorithm 1 / Algorithm 2 phase 2: category-based alpha selection.

The paper's rule (Appendix A.5, and lines 11-24 of Algorithm 2):

    for each category c:
        for each alpha in c:
            score = wc * confidence + wr * risk
            if score > threshold: keep it
        if any kept: select argmax(score) within the category

The category loop is the point -- it is what produces diversification across
momentum, value, volatility and so on rather than nine flavours of momentum.
It is also where Appendix A.3's duplicate formulas bite: `ROC` (momentum) and
`Momentum Oscillator` (momentum) are the same expression, and `ATR(14)` sits in
both Volatility and Technical, so "one best alpha per category" can and does
return the same signal twice. Selections are checked for that and the
collisions are reported rather than silently deduplicated -- the paper's rule is
what is being tested, not an improved version of it.

Both agents can be switched off independently for the Table 7/8 ablations. With
one agent disabled its weight is dropped and the other's score is used alone,
rather than being replaced by a constant, which would leave the threshold
comparing against a different scale.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.agents.confidence import score_confidence
from alpha_lab.agents.factor_stats import FactorStats
from alpha_lab.agents.risk import score_risk
from alpha_lab.alphas.registry import AlphaRegistry
from alpha_lab.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Selection:
    alphas: list[str]
    signs: pd.Series
    """alpha -> +1/-1 orientation from the confidence agent."""

    table: pd.DataFrame
    """Every candidate with its component scores, for audit."""

    as_of: pd.Timestamp
    duplicate_collisions: list[tuple[str, str]]
    """Pairs of selected alphas that A.3 defines identically."""

    @property
    def categories(self) -> list[str]:
        if self.table.empty:
            return []
        chosen = self.table[self.table["selected"]]
        return sorted(chosen["category"].unique().tolist())


def _percentile(values: pd.Series) -> pd.Series:
    """Rank into [0, 1] so confidence and risk are on a common scale."""
    return values.rank(pct=True)


def select_alphas(
    stats: FactorStats,
    registry: AlphaRegistry,
    cfg: Config,
    as_of: pd.Timestamp,
) -> Selection:
    """Run Algorithm 1 as of ``as_of``, using only information available then."""
    confidence = score_confidence(stats, cfg, as_of)
    risk = score_risk(stats, cfg, as_of)

    use_csa = cfg.selection.use_csa
    use_rpa = cfg.selection.use_rpa
    if not use_csa and not use_rpa:
        raise ValueError("at least one of use_csa / use_rpa must be enabled")

    # Percentile-rank each agent's output so a 0.6/0.4 blend means what it says.
    conf_pct = _percentile(confidence.score) if use_csa else None
    risk_pct = _percentile(risk.score) if use_rpa else None

    wc = cfg.selection.w_confidence if use_csa else 0.0
    wr = cfg.selection.w_risk if use_rpa else 0.0
    total = wc + wr
    if total <= 0:
        raise ValueError("w_confidence + w_risk must be positive for the enabled agents")

    combined = pd.Series(0.0, index=stats.alphas)
    if use_csa:
        combined = combined.add(wc * conf_pct, fill_value=np.nan)
    if use_rpa:
        combined = combined.add(wr * risk_pct, fill_value=np.nan)
    combined = combined / total

    # An alpha missing either enabled agent's score is unmeasured, not bad.
    measured = pd.Series(True, index=stats.alphas)
    if use_csa:
        measured &= confidence.score.notna()
    if use_rpa:
        measured &= risk.score.notna()
    combined = combined.where(measured)

    table = pd.DataFrame(
        {
            "alpha": stats.alphas,
            "category": [registry[a].category for a in stats.alphas],
            "confidence": confidence.score.reindex(stats.alphas),
            "confidence_pct": conf_pct.reindex(stats.alphas) if use_csa else np.nan,
            "mean_ic": confidence.raw_mean_ic.reindex(stats.alphas),
            "ic_sign": confidence.sign.reindex(stats.alphas),
            "n_ic_obs": confidence.n_obs.reindex(stats.alphas),
            "risk": risk.score.reindex(stats.alphas),
            "risk_pct": risk_pct.reindex(stats.alphas) if use_rpa else np.nan,
            "spread_vol": risk.volatility.reindex(stats.alphas),
            "spread_max_dd": risk.max_drawdown.reindex(stats.alphas),
            "ic_stability": risk.ic_stability.reindex(stats.alphas),
            "score": combined.reindex(stats.alphas),
        }
    ).set_index("alpha", drop=False)

    # Threshold, then argmax within each category (Algorithm 2, lines 16-22).
    # Note the threshold is a *percentile* level, because both agent scores were
    # percentile-ranked above to make their weighted sum coherent.
    eligible = table[table["score"].notna() & (table["score"] > cfg.selection.threshold)]
    selected: list[str] = []
    for category in sorted(eligible["category"].unique()):
        in_category = eligible[eligible["category"] == category]
        top = in_category.nlargest(cfg.selection.per_category, "score")
        selected.extend(top["alpha"].tolist())

    table["selected"] = table["alpha"].isin(selected)

    collisions: list[tuple[str, str]] = []
    for name in selected:
        dup = registry[name].duplicate_of
        if dup and dup in selected:
            pair = tuple(sorted((name, dup)))
            if pair not in collisions:
                collisions.append(pair)  # type: ignore[arg-type]
    if collisions:
        logger.info(
            "selection on %s picked duplicate A.3 formulas: %s",
            as_of.date(), collisions,
        )

    signs = confidence.sign.reindex(selected).fillna(1.0)
    return Selection(
        alphas=selected,
        signs=signs,
        table=table.sort_values("score", ascending=False),
        as_of=as_of,
        duplicate_collisions=collisions,
    )
