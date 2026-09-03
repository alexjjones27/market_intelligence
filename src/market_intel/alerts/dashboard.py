"""Research dashboard/export -- output #3 from the build spec: upcoming
event calendar, event-study history by event type, sector heat map,
macro regime state, and a flat CSV/parquet export for backtesting.
"""
from __future__ import annotations

import sqlite3
import statistics
from pathlib import Path

import pandas as pd

from market_intel.db.database import loads


def upcoming_event_calendar(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    earnings = conn.execute(
        "SELECT ticker, company, scheduled_at, fiscal_period, confirmed, is_mocked "
        "FROM earnings_calendar ORDER BY scheduled_at ASC LIMIT ?",
        (limit,),
    ).fetchall()
    macro = conn.execute(
        "SELECT release_name, series_id, scheduled_at, country FROM macro_calendar "
        "ORDER BY scheduled_at ASC LIMIT ?",
        (limit,),
    ).fetchall()

    rows = [
        {"type": "earnings", "label": f"{r['ticker']} earnings", "scheduled_at": r["scheduled_at"],
         "is_mocked": bool(r["is_mocked"])}
        for r in earnings
    ] + [
        {"type": "macro", "label": r["release_name"], "scheduled_at": r["scheduled_at"], "is_mocked": False}
        for r in macro
    ]
    rows.sort(key=lambda r: r["scheduled_at"])
    return rows


def event_study_by_type(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT event_type, interpretation, market_reaction FROM events").fetchall()
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["event_type"], []).append(
            {"interpretation": loads(r["interpretation"]), "market_reaction": loads(r["market_reaction"])}
        )

    summary = []
    for event_type, items in by_type.items():
        returns_1d = [i["market_reaction"].get("return_1d") for i in items if i["market_reaction"].get("return_1d") is not None]
        returns_5d = [i["market_reaction"].get("return_5d") for i in items if i["market_reaction"].get("return_5d") is not None]
        surprises = [i["interpretation"].get("surprise_score") for i in items if i["interpretation"].get("surprise_score") is not None]
        summary.append(
            {
                "event_type": event_type,
                "count": len(items),
                "avg_return_1d": round(statistics.mean(returns_1d), 4) if returns_1d else None,
                "avg_return_5d": round(statistics.mean(returns_5d), 4) if returns_5d else None,
                "avg_surprise_score": round(statistics.mean(surprises), 4) if surprises else None,
            }
        )
    summary.sort(key=lambda s: s["count"], reverse=True)
    return summary


def sector_heat_map(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT w.sector AS sector, e.market_reaction AS market_reaction
        FROM events e
        JOIN watchlist w ON w.ticker = e.ticker
        WHERE w.sector IS NOT NULL
        """
    ).fetchall()

    by_sector: dict[str, list[float]] = {}
    for r in rows:
        mr = loads(r["market_reaction"])
        ret = mr.get("return_1d")
        if ret is not None:
            by_sector.setdefault(r["sector"], []).append(ret)

    heat = [
        {"sector": sector, "avg_return_1d": round(statistics.mean(vals), 4), "n_events": len(vals)}
        for sector, vals in by_sector.items()
    ]
    heat.sort(key=lambda h: h["avg_return_1d"], reverse=True)
    return heat


def macro_regime_state(conn: sqlite3.Connection) -> dict | None:
    row = conn.execute(
        "SELECT * FROM macro_regime_history ORDER BY as_of DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    return {
        "as_of": row["as_of"],
        "growth_state": row["growth_state"],
        "inflation_state": row["inflation_state"],
        "regime_label": row["regime_label"],
        "expected_cross_asset_direction": loads(row["expected_cross_asset_direction"]),
    }


def export_events_table(conn: sqlite3.Connection) -> pd.DataFrame:
    rows = conn.execute("SELECT * FROM events").fetchall()
    flat_rows = []
    for r in rows:
        facts = loads(r["facts"])
        exposure = loads(r["exposure"])
        interp = loads(r["interpretation"])
        mr = loads(r["market_reaction"])
        flat_rows.append(
            {
                "event_id": r["event_id"],
                "event_type": r["event_type"],
                "company": r["company"],
                "ticker": r["ticker"],
                "timestamp_utc": r["timestamp_utc"],
                "source_tier": r["source_tier"],
                "eps_actual": facts.get("eps_actual"),
                "eps_consensus": facts.get("eps_consensus"),
                "revenue_actual": facts.get("revenue_actual"),
                "revenue_consensus": facts.get("revenue_consensus"),
                "guidance_direction": facts.get("guidance_direction"),
                "sectors": ",".join(exposure.get("sectors", [])),
                "surprise_score": interp.get("surprise_score"),
                "novelty_score": interp.get("novelty_score"),
                "market_sensitivity_score": interp.get("market_sensitivity_score"),
                "confirmation_score": interp.get("confirmation_score"),
                "confidence": interp.get("confidence"),
                "verification_flag": interp.get("verification_flag"),
                "return_5m": mr.get("return_5m"),
                "return_1h": mr.get("return_1h"),
                "return_1d": mr.get("return_1d"),
                "return_5d": mr.get("return_5d"),
                "volume_zscore": mr.get("volume_zscore"),
                "iv_change": mr.get("iv_change"),
            }
        )
    return pd.DataFrame(flat_rows)


def export_events_csv(conn: sqlite3.Connection, path: str | Path) -> Path:
    df = export_events_table(conn)
    path = Path(path)
    df.to_csv(path, index=False)
    return path


def export_events_parquet(conn: sqlite3.Connection, path: str | Path) -> Path:
    df = export_events_table(conn)
    path = Path(path)
    df.to_parquet(path, index=False)
    return path
