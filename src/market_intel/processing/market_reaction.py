"""Turns cached price_snapshots into a populated MarketReaction:
return_5m/1h/1d/5d, volume z-score, and (via the abnormal-return helper)
a benchmark-relative return for scoring.

Async-by-nature: any horizon whose price data isn't available yet (or
never was, e.g. a >60-day-old event past Yahoo's intraday retention
window) is left None rather than guessed at.
"""
from __future__ import annotations

import sqlite3
from datetime import timedelta

from market_intel.models.event import MarketReaction
from market_intel.processing.timestamps import parse_utc, to_utc_iso
from market_intel.scoring.market_confirmation import simple_return, volume_zscore

HORIZON_OFFSETS: dict[str, timedelta] = {
    "return_5m": timedelta(minutes=5),
    "return_1h": timedelta(hours=1),
    "return_1d": timedelta(days=1),
    "return_5d": timedelta(days=5),
}

HORIZON_TOLERANCE: dict[str, timedelta] = {
    "return_5m": timedelta(minutes=10),
    "return_1h": timedelta(minutes=20),
    "return_1d": timedelta(hours=16),
    "return_5d": timedelta(hours=16),
}

BASELINE_TOLERANCE = timedelta(minutes=30)


def nearest_snapshot(
    conn: sqlite3.Connection, ticker: str, target_utc: str, max_gap: timedelta
) -> sqlite3.Row | None:
    target_dt = parse_utc(target_utc)
    lo = to_utc_iso(target_dt - max_gap)
    hi = to_utc_iso(target_dt + max_gap)
    rows = conn.execute(
        "SELECT timestamp_utc, price, volume FROM price_snapshots "
        "WHERE ticker = ? AND timestamp_utc BETWEEN ? AND ? ORDER BY timestamp_utc",
        (ticker, lo, hi),
    ).fetchall()
    if not rows:
        return None
    return min(rows, key=lambda r: abs((parse_utc(r["timestamp_utc"]) - target_dt).total_seconds()))


def compute_returns(conn: sqlite3.Connection, ticker: str, event_ts_utc: str) -> dict[str, float | None]:
    baseline = nearest_snapshot(conn, ticker, event_ts_utc, BASELINE_TOLERANCE)
    if baseline is None:
        return {k: None for k in HORIZON_OFFSETS}

    base_price = baseline["price"]
    event_dt = parse_utc(event_ts_utc)

    out: dict[str, float | None] = {}
    for field_name, offset in HORIZON_OFFSETS.items():
        target = to_utc_iso(event_dt + offset)
        snap = nearest_snapshot(conn, ticker, target, HORIZON_TOLERANCE[field_name])
        out[field_name] = simple_return(base_price, snap["price"]) if snap else None
    return out


def compute_volume_zscore_for_event(
    conn: sqlite3.Connection, ticker: str, event_ts_utc: str, trailing_days: int = 20
) -> float | None:
    event_date = parse_utc(event_ts_utc).date()
    rows = conn.execute(
        "SELECT timestamp_utc, volume FROM price_snapshots "
        "WHERE ticker = ? AND granularity = 'daily' AND volume IS NOT NULL "
        "ORDER BY timestamp_utc",
        (ticker,),
    ).fetchall()
    if not rows:
        return None

    event_day_volume = None
    history: list[float] = []
    for r in rows:
        row_date = parse_utc(r["timestamp_utc"]).date()
        if row_date == event_date:
            event_day_volume = r["volume"]
        elif row_date < event_date:
            history.append(r["volume"])

    history = history[-trailing_days:]
    return volume_zscore(event_day_volume, history)


def daily_return_for_date(conn: sqlite3.Connection, ticker: str, event_date) -> float | None:
    """Close-to-close return for the daily bar on/after `event_date` vs.
    the trading day immediately before it. Used where we only have daily
    (not intraday) granularity for a ticker, e.g. peers in a sector
    breadth calculation."""
    rows = conn.execute(
        "SELECT timestamp_utc, price FROM price_snapshots WHERE ticker = ? AND granularity = 'daily' "
        "ORDER BY timestamp_utc",
        (ticker,),
    ).fetchall()
    if len(rows) < 2:
        return None
    bars = [(parse_utc(r["timestamp_utc"]).date(), r["price"]) for r in rows]
    idx = next((i for i, (d, _) in enumerate(bars) if d >= event_date), None)
    if idx is None or idx == 0:
        return None
    return simple_return(bars[idx - 1][1], bars[idx][1])


def populate_market_reaction(conn: sqlite3.Connection, ticker: str, event_ts_utc: str) -> MarketReaction:
    returns = compute_returns(conn, ticker, event_ts_utc)
    vz = compute_volume_zscore_for_event(conn, ticker, event_ts_utc)
    return MarketReaction(
        return_5m=returns["return_5m"],
        return_1h=returns["return_1h"],
        return_1d=returns["return_1d"],
        return_5d=returns["return_5d"],
        volume_zscore=vz,
        iv_change=None,  # filled by the options provider separately -- see ingestion/market_data.py
    )
