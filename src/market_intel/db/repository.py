"""Event/evidence/raw-item persistence and the read queries scoring
needs (prior events for novelty, trailing earnings history for
surprise z-scores). Split out from database.py to keep that module to
pure connection/schema/seed concerns.
"""
from __future__ import annotations

import sqlite3
import uuid

from market_intel.db.database import dumps, loads, row_to_dict
from market_intel.models.event import Event
from market_intel.models.raw_item import RawItem
from market_intel.scoring.surprise import raw_surprise


def insert_raw_item(conn: sqlite3.Connection, item: RawItem) -> None:
    """Idempotent: a re-run that re-fetches the same (ticker, source,
    url) is ignored rather than accumulating a duplicate row -- see
    schema.sql's UNIQUE constraint on raw_items."""
    conn.execute(
        """
        INSERT OR IGNORE INTO raw_items (raw_item_id, source, source_tier, publisher, url, ticker_guess,
            company_guess, title, body_text, event_type_guess, published_at, event_timestamp_guess,
            exchange_local_time, retrieved_at, confirmed, content_hash, is_mocked)
        VALUES (:raw_item_id, :source, :source_tier, :publisher, :url, :ticker_guess, :company_guess,
            :title, :body_text, :event_type_guess, :published_at, :event_timestamp_guess,
            :exchange_local_time, :retrieved_at, :confirmed, :content_hash, :is_mocked)
        """,
        item.to_row(),
    )


def insert_event(conn: sqlite3.Connection, event: Event) -> None:
    """Upsert, keyed on event_id. event_id is deterministic (see
    dedup.py:_stable_event_id) so a re-run against the same underlying
    disclosure updates this row in place instead of accumulating a
    duplicate event -- evidence is fully replaced on each upsert rather
    than appended, so corroborating-source counts stay accurate instead
    of double-counting across runs."""
    row = event.to_row()
    conn.execute(
        """
        INSERT INTO events (event_id, event_type, company, ticker, timestamp_utc, exchange_local_time,
            source_tier, facts, exposure, interpretation, market_reaction, data_quality)
        VALUES (:event_id, :event_type, :company, :ticker, :timestamp_utc, :exchange_local_time,
            :source_tier, :facts, :exposure, :interpretation, :market_reaction, :data_quality)
        ON CONFLICT(event_id) DO UPDATE SET
            event_type = excluded.event_type,
            company = excluded.company,
            ticker = excluded.ticker,
            timestamp_utc = excluded.timestamp_utc,
            exchange_local_time = excluded.exchange_local_time,
            source_tier = excluded.source_tier,
            facts = excluded.facts,
            exposure = excluded.exposure,
            interpretation = excluded.interpretation,
            market_reaction = excluded.market_reaction,
            data_quality = excluded.data_quality,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        """,
        {
            **row,
            "facts": dumps(row["facts"]),
            "exposure": dumps(row["exposure"]),
            "interpretation": dumps(row["interpretation"]),
            "market_reaction": dumps(row["market_reaction"]),
            "data_quality": dumps(row["data_quality"]),
        },
    )
    conn.execute("DELETE FROM evidence WHERE event_id = ?", (event.event_id,))
    for ev in event.evidence:
        conn.execute(
            """
            INSERT INTO evidence (evidence_id, event_id, source, url, publisher, published_at,
                retrieved_at, confirmed, is_mocked, raw_item_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()), event.event_id, ev.source, ev.url, ev.publisher,
                ev.published_at, ev.retrieved_at, int(ev.confirmed), int(ev.is_mocked), ev.raw_item_id,
            ),
        )


def update_event_scoring(conn: sqlite3.Connection, event: Event) -> None:
    row = event.to_row()
    conn.execute(
        """
        UPDATE events
        SET facts = ?, exposure = ?, interpretation = ?, market_reaction = ?, data_quality = ?,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE event_id = ?
        """,
        (
            dumps(row["facts"]), dumps(row["exposure"]), dumps(row["interpretation"]),
            dumps(row["market_reaction"]), dumps(row["data_quality"]), event.event_id,
        ),
    )


def _row_to_event(conn: sqlite3.Connection, row: sqlite3.Row) -> Event:
    d = row_to_dict(row)
    d["facts"] = loads(d["facts"])
    d["exposure"] = loads(d["exposure"])
    d["interpretation"] = loads(d["interpretation"])
    d["market_reaction"] = loads(d["market_reaction"])
    d["data_quality"] = loads(d.get("data_quality"))
    evidence_rows = conn.execute(
        "SELECT source, url, publisher, published_at, retrieved_at, confirmed, is_mocked, raw_item_id "
        "FROM evidence WHERE event_id = ?",
        (row["event_id"],),
    ).fetchall()
    evidence = []
    for er in evidence_rows:
        ed = row_to_dict(er)
        ed["confirmed"] = bool(ed["confirmed"])
        ed["is_mocked"] = bool(ed["is_mocked"])
        evidence.append(ed)
    return Event.from_row(d, evidence)


def get_prior_events(
    conn: sqlite3.Connection, ticker: str, event_type: str, before_timestamp_utc: str, limit: int = 5
) -> list[Event]:
    rows = conn.execute(
        "SELECT * FROM events WHERE ticker = ? AND event_type = ? AND timestamp_utc < ? "
        "ORDER BY timestamp_utc DESC LIMIT ?",
        (ticker, event_type, before_timestamp_utc, limit),
    ).fetchall()
    return [_row_to_event(conn, r) for r in rows]


def get_earnings_surprise_history(
    conn: sqlite3.Connection, ticker: str, metric: str, exclude_period: str | None = None, limit: int = 8
) -> list[float]:
    """metric: 'eps' or 'revenue'. Raw surprises computed from trailing
    earnings_consensus rows, most recent first, excluding the period
    currently being scored (so a company can't be standardized against
    its own current-quarter number)."""
    rows = conn.execute(
        "SELECT fiscal_period, eps_actual, eps_consensus, revenue_actual, revenue_consensus "
        "FROM earnings_consensus WHERE ticker = ? ORDER BY first_captured_at DESC LIMIT ?",
        (ticker, limit + 1),
    ).fetchall()
    out: list[float] = []
    for r in rows:
        if exclude_period and r["fiscal_period"] == exclude_period:
            continue
        s = raw_surprise(r["eps_actual"], r["eps_consensus"]) if metric == "eps" else raw_surprise(
            r["revenue_actual"], r["revenue_consensus"]
        )
        if s is not None:
            out.append(s)
    return out[:limit]
