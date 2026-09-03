"""Earnings + macro calendar seeding (feeds the research dashboard's
"upcoming calendar" panel).

*** EARNINGS CALENDAR IS STUBBED / MOCKED ***. No forward-looking
earnings-date provider is wired up. Real next-earnings dates would
naturally come from the same vendor as ingestion/earnings.py --
Finnhub's `/calendar/earnings` is a good fit since it shares the news
API key already in use.

*** MACRO CALENDAR IS A HAND-MAINTAINED CADENCE HEURISTIC, NOT A REAL
CALENDAR FEED ***. It assumes each series in ingestion/macro.py's
MACRO_SERIES releases roughly monthly (quarterly for GDP) and picks a
pseudo-random day within that window. A real implementation should
consume BLS/BEA/Fed's published release calendars directly instead of
guessing cadence.
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone

from market_intel.ingestion.macro import MACRO_SERIES

QUARTERLY_SERIES = {"GDP"}
ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def mock_next_earnings_date(ticker: str, as_of: datetime | None = None) -> datetime:
    as_of = as_of or datetime.now(timezone.utc)
    seed = int(hashlib.sha256(ticker.encode()).hexdigest(), 16)
    days_out = 10 + (seed % 80)  # 10-90 days out
    return as_of + timedelta(days=days_out)


def seed_earnings_calendar(conn: sqlite3.Connection, tickers: list[str], watchlist: dict[str, dict]) -> None:
    if not tickers:
        return
    conn.execute(
        f"DELETE FROM earnings_calendar WHERE ticker IN ({','.join('?' for _ in tickers)})", tickers
    )
    for ticker in tickers:
        company = watchlist.get(ticker, {}).get("company", ticker)
        scheduled = mock_next_earnings_date(ticker)
        conn.execute(
            "INSERT INTO earnings_calendar (ticker, company, scheduled_at, fiscal_period, confirmed, is_mocked) "
            "VALUES (?, ?, ?, NULL, 0, 1)",
            (ticker, company, scheduled.strftime(ISO_FMT)),
        )


def seed_macro_calendar(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM macro_calendar")
    now = datetime.now(timezone.utc)
    for name, series_id in MACRO_SERIES.items():
        cadence_days = 90 if name in QUARTERLY_SERIES else 30
        seed = int(hashlib.sha256(name.encode()).hexdigest(), 16)
        days_out = (seed % cadence_days) + 1
        scheduled = now + timedelta(days=days_out)
        conn.execute(
            "INSERT INTO macro_calendar (release_name, series_id, scheduled_at, country) VALUES (?, ?, ?, ?)",
            (name, series_id, scheduled.strftime(ISO_FMT), "United States"),
        )
