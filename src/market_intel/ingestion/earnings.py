"""Earnings consensus (EPS/revenue actual vs. consensus) ingestion.

*** STUBBED / MOCKED -- NO REAL PROVIDER WIRED UP. ***

The build spec says: "ask which data provider I have access to if not
obvious; otherwise stub the interface and mock the response." No provider
was specified, so this module ships a `MockEarningsProvider` behind the
`EarningsConsensusProvider` interface. Every value it returns is
synthetic and every row persisted to `earnings_consensus` is flagged
`is_mocked=1`.

TO WIRE UP A REAL PROVIDER: implement `EarningsConsensusProvider` with a
real backend and set `EARNINGS_PROVIDER` in .env to something other than
"mock" (see config.py). Reasonable candidates that expose actual vs.
consensus EPS/revenue:
    - Finnhub: /stock/earnings (surprise history) + /calendar/earnings
    - Alpha Vantage: EARNINGS / EARNINGS_ESTIMATES
    - Zacks, IEX Cloud, or another paid vendor

Point-in-time discipline: whichever provider is wired up must write the
FIRST-SEEN consensus estimate once and never overwrite it (revisions to
consensus estimates happen constantly pre-earnings; backtests need the
number that was live at alert time, not the latest one). See
`db/schema.sql: earnings_consensus` and `evaluation/` for how this is
enforced -- rows are insert-only, keyed by (ticker, fiscal_period,
first_captured_at).
"""
from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from market_intel.config import settings

logger = logging.getLogger(__name__)


class EarningsConsensusProvider(ABC):
    is_mocked: bool = True

    @abstractmethod
    def get_consensus(self, ticker: str, fiscal_period: str) -> dict | None:
        """Return {'eps_consensus': float, 'revenue_consensus': float,
        'eps_actual': float|None, 'revenue_actual': float|None,
        'source': str} or None if unavailable."""
        raise NotImplementedError


class MockEarningsProvider(EarningsConsensusProvider):
    """Deterministic synthetic consensus/actuals, seeded off ticker +
    fiscal_period so repeated calls are stable (useful for tests)."""

    is_mocked = True

    def get_consensus(self, ticker: str, fiscal_period: str) -> dict | None:
        seed = int(hashlib.sha256(f"{ticker}|{fiscal_period}".encode()).hexdigest(), 16)
        eps_consensus = round(0.5 + (seed % 400) / 100, 2)              # 0.50 - 4.49
        surprise_pct = ((seed // 400) % 41 - 20) / 100                  # -0.20 .. +0.20
        eps_actual = round(eps_consensus * (1 + surprise_pct), 2)

        revenue_consensus = round(1_000_000_000 + (seed % 50) * 250_000_000, 0)
        rev_surprise_pct = ((seed // 20000) % 21 - 10) / 100            # -0.10 .. +0.10
        revenue_actual = round(revenue_consensus * (1 + rev_surprise_pct), 0)

        logger.info(
            "[MOCK] earnings consensus for %s %s (eps_consensus=%.2f, eps_actual=%.2f)",
            ticker, fiscal_period, eps_consensus, eps_actual,
        )
        return {
            "eps_consensus": eps_consensus,
            "eps_actual": eps_actual,
            "revenue_consensus": revenue_consensus,
            "revenue_actual": revenue_actual,
            "source": "mock",
        }


def get_provider() -> EarningsConsensusProvider:
    if settings.earnings_is_mocked:
        return MockEarningsProvider()
    raise NotImplementedError(
        f"EARNINGS_PROVIDER={settings.earnings_provider!r} is not wired up yet. "
        "Implement EarningsConsensusProvider for it in ingestion/earnings.py."
    )


def current_fiscal_period(as_of: datetime | None = None) -> str:
    """Best-effort calendar-quarter label, e.g. '2026Q3'. Real fiscal
    calendars vary by company; this is a simplification for the MVP mock."""
    dt = as_of or datetime.now(timezone.utc)
    quarter = (dt.month - 1) // 3 + 1
    return f"{dt.year}Q{quarter}"


def previous_fiscal_periods(period: str, n: int) -> list[str]:
    """['2026Q3', ...n back] -> ['2026Q2', '2026Q1', '2025Q4', ...]."""
    year, quarter = int(period[:4]), int(period[5])
    out = []
    for _ in range(n):
        quarter -= 1
        if quarter == 0:
            quarter, year = 4, year - 1
        out.append(f"{year}Q{quarter}")
    return out
