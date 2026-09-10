"""Configuration for the alpha_lab replication pipeline.

Everything a user is likely to want to sweep -- top-k/drop-n, the CSA/RPA
weights, the IC horizon, the walk-forward window geometry, transaction costs --
lives here and is loaded from YAML. Core logic never hard-codes these.

Load with :func:`load_config`; see ``configs/default.yaml``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class DataConfig:
    """Where market data comes from and over what span."""

    start: str = "2018-01-01"
    """Data pull start. Deliberately earlier than ``backtest_start`` so the
    longest-lookback alphas (50-day STD, 26-day EMA) are warm by the time the
    first backtest date arrives. Burn-in is dropped, never back-filled."""

    end: str = "2024-06-30"
    backtest_start: str = "2019-01-01"

    cache_dir: str = "data/cache"
    price_source: Literal["yahoo"] = "yahoo"
    fundamentals_source: Literal["sec_edgar", "none"] = "sec_edgar"
    macro_source: Literal["fred", "none"] = "fred"

    max_universe_size: int | None = None
    """Cap on distinct tickers pulled (None = all). Used to keep smoke runs fast;
    a cap biases the universe, so runs that use one are flagged in the report."""

    sec_user_agent: str = "alpha_lab research (contact@example.com)"
    request_sleep_s: float = 0.12
    """SEC asks for <=10 req/s. Applied to every SEC call."""

    min_price: float = 3.0
    """Drop stock-days under this price. Penny-stock noise dominates ranking
    signals and is not realistically tradeable at the sizes implied here."""

    min_dollar_volume: float = 1_000_000.0


@dataclass(frozen=True)
class UniverseConfig:
    """Point-in-time index membership."""

    name: str = "sp500"
    point_in_time: bool = True
    """If True, membership is reconstructed backwards from the current
    constituent list using the dated add/drop changelog, so a stock only enters
    the tradeable set on the day it actually joined the index. If False, the
    *current* list is used for all history -- which is survivorship bias, and
    the report says so loudly."""

    membership_buffer_days: int = 0


@dataclass(frozen=True)
class AlphaConfig:
    """Stage 1: Seed Alpha Factory."""

    categories: list[str] = field(
        default_factory=lambda: [
            "momentum",
            "mean_reversion",
            "volatility",
            "fundamental",
            "liquidity",
            "quality",
            "growth",
            "technical",
            "macro",
        ]
    )
    include_unavailable: bool = False
    """Appendix A.3 lists a few alphas we cannot compute from daily OHLCV +
    XBRL (bid-ask spread needs quote data). Off by default; when on they are
    emitted as all-NaN so the count matches the paper's table."""

    winsorize_quantile: float = 0.01
    """Cross-sectional winsorisation per day before normalisation. 0 disables."""

    cross_sectional_normalize: Literal["zscore", "rank", "none"] = "zscore"
    """Alphas in A.3 are wildly different scales (VOLUME*CLOSE is ~1e9,
    EPS/CLOSE is ~0.05). The MLP needs them on a common scale, and the paper's
    own combination step implicitly assumes it. Applied per-day, cross-
    sectionally, so it never mixes information across time."""

    min_cross_section: int = 20
    """Days with fewer valid names than this are dropped from normalisation."""

    fundamental_growth_lag_days: int = 252
    """A.3's growth factors are written ``X / DELAY(X, 1) - 1``. In a *daily*
    panel a one-step delay is one trading day, over which a quarterly-updating
    fundamental has not moved at all, so the factor would be identically zero.
    ``DELAY(X, 1)`` must mean one reporting period. We use 252 trading days --
    year-over-year growth of the trailing-twelve-month figure -- which is both
    the standard construction and immune to fiscal seasonality. Set to 63 for a
    quarter-over-quarter reading instead."""

    earnings_stability_window: int = 315
    """A.3 writes ``STD(EPS, 5) / MEAN(EPS, 5)`` without units. Five daily bars
    of a quarterly-updating TTM series is a constant (std = 0), so 5 must mean
    five reporting periods; 315 trading days is five quarters."""


@dataclass(frozen=True)
class CSAConfig:
    """Stage 2: Confidence Score Agent.

    The paper defines confidence only as ``E[IC(alpha | market state)]``. We
    operationalise that as a rolling *out-of-sample* rank IC: at every scoring
    date, IC is measured over a trailing window that ends far enough back that
    the forward return used to compute it was already observable. See
    ``agents/confidence.py`` for the exact lag arithmetic.
    """

    ic_horizon: int = 5
    """Forward-return horizon N (trading days) for the IC."""

    method: Literal["spearman", "pearson"] = "spearman"
    window: Literal["rolling", "expanding"] = "rolling"
    window_days: int = 252
    min_periods: int = 60
    confidence_threshold: float = 0.0
    """Algorithm 1's threshold X, applied to the *combined* score."""


@dataclass(frozen=True)
class RPAConfig:
    """Stage 3 of the agent layer: Risk Preference Agent.

    The paper gives only ``rho = f_risk(alpha, market state)``. We define
    f_risk explicitly as a blend of three penalties, each computed on the same
    trailing, already-observable window as the CSA.
    """

    lookback_days: int = 252
    min_periods: int = 60
    spread_quantile: float = 0.2
    """Long-short spread uses top/bottom quintile by default (0.2)."""

    w_volatility: float = 0.34
    w_drawdown: float = 0.33
    w_ic_stability: float = 0.33
    ic_stability_window: int = 63
    """Sub-window length used to measure whether the IC sign is stable."""


@dataclass(frozen=True)
class SelectionConfig:
    """Algorithm 1: category-based alpha selection."""

    w_confidence: float = 0.6
    w_risk: float = 0.4
    threshold: float = 0.0
    per_category: int = 1
    """Algorithm 2 line 21 takes argmax within each category (1 per category).
    Configurable so the category-diversification claim can be tested."""

    use_csa: bool = True
    use_rpa: bool = True
    """Ablation switches for Table 7/8."""


@dataclass(frozen=True)
class MLPConfig:
    """Stage 3: weight optimisation. Paper Table 10's optimal row."""

    hidden_nodes: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    weight_decay: float = 0.001
    max_epochs: int = 200
    patience: int = 20
    seed: int = 17
    n_restarts: int = 3
    """Median-of-restarts to keep results from hanging on one lucky init."""

    target_horizon: int = 5
    """Forward return the MLP predicts. Defaults to the CSA horizon."""

    standardize_target: bool = True


@dataclass(frozen=True)
class PortfolioConfig:
    """Appendix A.8: top-k / drop-n daily rebalancing."""

    top_k: int = 13
    drop_n: int = 5
    weighting: Literal["equal"] = "equal"
    cost_bps: float = 5.0
    """One-way transaction cost in basis points, applied to traded notional."""

    slippage_bps: float = 5.0
    """Additional one-way slippage. Total one-way friction = cost + slippage."""

    execution_lag_days: int = 1
    """Signal computed from data through close of t is executed at t+1's close.
    Setting this to 0 reproduces the (unrealistic) same-bar-execution
    assumption; it exists so the cost of that assumption can be measured."""


@dataclass(frozen=True)
class WalkForwardConfig:
    """Table 5 geometry, generalised to a rolling walk-forward."""

    train_months: int = 18
    validation_months: int = 6
    test_months: int = 6
    step_months: int = 6
    start: str = "2019-01-01"
    end: str = "2024-06-30"
    paper_replication_folds: bool = True
    """Also emit the three literal Table 5 SP500 folds for direct comparison."""


@dataclass(frozen=True)
class EvaluationConfig:
    risk_free_rate: float = 0.02
    trading_days: int = 252
    bootstrap_iterations: int = 2000
    bootstrap_block_size: int = 21
    """Stationary/block bootstrap block length, in trading days. Daily strategy
    returns are autocorrelated; an iid bootstrap would understate the variance."""

    hac_max_lags: int | None = None
    seed: int = 17


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    alphas: AlphaConfig = field(default_factory=AlphaConfig)
    csa: CSAConfig = field(default_factory=CSAConfig)
    rpa: RPAConfig = field(default_factory=RPAConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def cache_path(self) -> Path:
        p = Path(self.data.cache_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def replace(self, **section_overrides: dict[str, Any]) -> "Config":
        """Return a copy with nested overrides, e.g.
        ``cfg.replace(portfolio={"top_k": 20})``. Used by the sweep drivers so
        sensitivity runs never mutate shared state."""
        updates: dict[str, Any] = {}
        for section, values in section_overrides.items():
            current = getattr(self, section)
            updates[section] = dataclasses.replace(current, **values)
        return dataclasses.replace(self, **updates)


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "universe": UniverseConfig,
    "alphas": AlphaConfig,
    "csa": CSAConfig,
    "rpa": RPAConfig,
    "selection": SelectionConfig,
    "mlp": MLPConfig,
    "portfolio": PortfolioConfig,
    "walkforward": WalkForwardConfig,
    "evaluation": EvaluationConfig,
}


def load_config(path: str | Path | None = None) -> Config:
    """Load a Config from YAML. Unknown keys raise rather than being ignored --
    a silently-dropped ``top_k`` would invalidate a whole sweep."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}

    unknown_sections = set(raw) - set(_SECTIONS)
    if unknown_sections:
        raise ValueError(f"unknown config section(s): {sorted(unknown_sections)}")

    sections: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        values = raw.get(name) or {}
        if not isinstance(values, dict):
            raise ValueError(f"config section '{name}' must be a mapping")
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = set(values) - valid
        if unknown:
            raise ValueError(f"unknown key(s) in '{name}': {sorted(unknown)}")
        sections[name] = cls(**values)
    return Config(**sections)
