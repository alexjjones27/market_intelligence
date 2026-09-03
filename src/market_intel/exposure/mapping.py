"""Cross-asset / read-through engine.

Static ticker -> sector -> {ETFs, peers, commodities, countries, rate &
FX sensitivity} mapping, per the build spec: "Start with a static
mapping table that's easy to extend, rather than trying to infer
relationships dynamically in v1." Backed by `data/sector_map.json`
(seeded into the `sector_map` / `watchlist` tables -- see
db/database.py:seed_static_tables).

Extend by adding a sector block (or a ticker under an existing sector's
`members`) to data/sector_map.json and re-running the seed step; no code
change needed.
"""
from __future__ import annotations

import sqlite3

from market_intel.db.database import loads
from market_intel.models.event import Exposure


def get_exposure_for_ticker(ticker: str, conn: sqlite3.Connection) -> Exposure:
    watch_row = conn.execute("SELECT sector FROM watchlist WHERE ticker = ?", (ticker,)).fetchone()
    sector = watch_row["sector"] if watch_row else None
    if not sector:
        return Exposure()

    sector_row = conn.execute("SELECT * FROM sector_map WHERE sector = ?", (sector,)).fetchone()
    if not sector_row:
        return Exposure(sectors=[sector])

    members = loads(sector_row["members"])
    member = members.get(ticker, {})

    return Exposure(
        sectors=[sector],
        countries=member.get("countries", []),
        commodities=loads(sector_row["commodities"]),
        peers=member.get("peers", []),
        sector_etfs=loads(sector_row["sector_etfs"]),
    )


def get_sector_rate_sensitivity(ticker: str, conn: sqlite3.Connection) -> str | None:
    watch_row = conn.execute("SELECT sector FROM watchlist WHERE ticker = ?", (ticker,)).fetchone()
    if not watch_row or not watch_row["sector"]:
        return None
    sector_row = conn.execute(
        "SELECT rate_sensitivity FROM sector_map WHERE sector = ?", (watch_row["sector"],)
    ).fetchone()
    return sector_row["rate_sensitivity"] if sector_row else None


def get_fx_sensitivity(ticker: str, conn: sqlite3.Connection) -> list[str]:
    watch_row = conn.execute("SELECT sector FROM watchlist WHERE ticker = ?", (ticker,)).fetchone()
    if not watch_row or not watch_row["sector"]:
        return []
    sector_row = conn.execute(
        "SELECT fx_sensitivity FROM sector_map WHERE sector = ?", (watch_row["sector"],)
    ).fetchone()
    return loads(sector_row["fx_sensitivity"]) if sector_row else []


def get_sector_etfs(ticker: str, conn: sqlite3.Connection) -> list[str]:
    return get_exposure_for_ticker(ticker, conn).sector_etfs


def get_peers(ticker: str, conn: sqlite3.Connection) -> list[str]:
    return get_exposure_for_ticker(ticker, conn).peers
