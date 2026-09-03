"""SQLite connection + schema bootstrap + small helper functions.

Kept intentionally thin: this is a data-access layer, not an ORM. Callers
build dicts / JSON blobs; this module just knows how to persist and fetch
them. All timestamps stored as UTC ISO-8601 strings.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from market_intel.config import settings

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or settings.db_path
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | None = None) -> None:
    conn = get_connection(db_path)
    try:
        schema_sql = SCHEMA_PATH.read_text()
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def connect(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)


def loads(s: str | None) -> Any:
    if not s:
        return {}
    return json.loads(s)


def seed_static_tables(db_path: str | None = None) -> None:
    """Load watchlist.json / sector_map.json into the DB. Idempotent (upsert)."""
    watchlist = json.loads(Path(settings.watchlist_path).read_text())
    sector_map = json.loads(Path(settings.sector_map_path).read_text())

    with connect(db_path) as conn:
        for entry in watchlist.get("tickers", []):
            conn.execute(
                """
                INSERT INTO watchlist (ticker, company, cik, sector, active)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(ticker) DO UPDATE SET
                    company=excluded.company,
                    cik=excluded.cik,
                    sector=excluded.sector,
                    active=1
                """,
                (entry["ticker"], entry["company"], entry.get("cik"), entry.get("sector")),
            )

        for sector, spec in sector_map.items():
            if sector.startswith("_"):
                continue
            conn.execute(
                """
                INSERT INTO sector_map (sector, sector_etfs, commodities, rate_sensitivity, fx_sensitivity, members)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(sector) DO UPDATE SET
                    sector_etfs=excluded.sector_etfs,
                    commodities=excluded.commodities,
                    rate_sensitivity=excluded.rate_sensitivity,
                    fx_sensitivity=excluded.fx_sensitivity,
                    members=excluded.members
                """,
                (
                    sector,
                    dumps(spec.get("sector_etfs", [])),
                    dumps(spec.get("commodities", [])),
                    spec.get("rate_sensitivity"),
                    dumps(spec.get("fx_sensitivity", [])),
                    dumps(spec.get("members", {})),
                ),
            )
