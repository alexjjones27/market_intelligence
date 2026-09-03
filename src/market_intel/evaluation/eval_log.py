"""Evaluation framework -- built from day one per the spec, even before
there's historical data to backfill.

Every alert generated gets a row in `alert_evaluation`, created at alert
time with just latency filled in (everything else starts NULL) and
updated asynchronously as outcomes become observable:

    log -> alerts (immutable, what we said and when)
    log -> alert_evaluation (mutable, filled in as reality unfolds)

Point-in-time discipline: alerts and their originating consensus/price
data are never rewritten after the fact -- only the *evaluation* row is
updated, and only with newly-observed outcomes, never by changing what
was known at alert time. `holdout` marks alerts that must be excluded
from any threshold-tuning process (spec: "use a holdout period for any
threshold tuning").
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def log_alert_created(conn: sqlite3.Connection, alert_id: str, latency_seconds: float | None, holdout: bool = False) -> None:
    conn.execute(
        """
        INSERT INTO alert_evaluation (alert_id, latency_seconds, holdout)
        VALUES (?, ?, ?)
        ON CONFLICT(alert_id) DO UPDATE SET latency_seconds = excluded.latency_seconds
        """,
        (alert_id, latency_seconds, int(holdout)),
    )


def record_outcome(
    conn: sqlite3.Connection,
    alert_id: str,
    return_5m: float | None = None,
    return_30m: float | None = None,
    return_1d: float | None = None,
    return_5d: float | None = None,
    volume_zscore_response: float | None = None,
    iv_change_response: float | None = None,
    sector_breadth: float | None = None,
    precision_label: str | None = None,   # "material" | "not_material" | "unknown"
    false_positive: bool | None = None,
    spread_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> None:
    pnl_bps = None
    if return_1d is not None:
        pnl_bps = compute_pnl_net_of_costs_bps(return_1d, spread_bps, slippage_bps)

    conn.execute(
        """
        UPDATE alert_evaluation
        SET return_5m = COALESCE(?, return_5m),
            return_30m = COALESCE(?, return_30m),
            return_1d = COALESCE(?, return_1d),
            return_5d = COALESCE(?, return_5d),
            volume_zscore_response = COALESCE(?, volume_zscore_response),
            iv_change_response = COALESCE(?, iv_change_response),
            sector_breadth = COALESCE(?, sector_breadth),
            precision_label = COALESCE(?, precision_label),
            false_positive = COALESCE(?, false_positive),
            pnl_net_of_costs_bps = COALESCE(?, pnl_net_of_costs_bps),
            evaluated_at = ?
        WHERE alert_id = ?
        """,
        (
            return_5m, return_30m, return_1d, return_5d,
            volume_zscore_response, iv_change_response, sector_breadth,
            precision_label, None if false_positive is None else int(false_positive),
            pnl_bps, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            alert_id,
        ),
    )


def compute_pnl_net_of_costs_bps(return_pct: float, spread_bps: float = 5.0, slippage_bps: float = 5.0) -> float:
    """Simple assumed cost model: gross return in bps minus a flat
    spread + slippage assumption. Not a real transaction-cost model --
    good enough to stop a backtest from silently ignoring costs, which
    the spec explicitly calls out."""
    return round(return_pct * 10_000 - spread_bps - slippage_bps, 2)


def precision(conn: sqlite3.Connection, holdout: bool | None = None) -> float | None:
    query = "SELECT false_positive FROM alert_evaluation WHERE false_positive IS NOT NULL"
    params: tuple = ()
    if holdout is not None:
        query += " AND holdout = ?"
        params = (int(holdout),)
    rows = conn.execute(query, params).fetchall()
    if not rows:
        return None
    fp = sum(r["false_positive"] for r in rows)
    return round(1 - fp / len(rows), 4)


def false_positive_rate(conn: sqlite3.Connection, holdout: bool | None = None) -> float | None:
    p = precision(conn, holdout)
    return None if p is None else round(1 - p, 4)


def average_latency_seconds(conn: sqlite3.Connection) -> float | None:
    row = conn.execute("SELECT AVG(latency_seconds) AS avg_latency FROM alert_evaluation WHERE latency_seconds IS NOT NULL").fetchone()
    return row["avg_latency"] if row and row["avg_latency"] is not None else None


def average_return_by_horizon(conn: sqlite3.Connection) -> dict[str, float | None]:
    row = conn.execute(
        """
        SELECT AVG(return_5m) AS r5m, AVG(return_30m) AS r30m, AVG(return_1d) AS r1d, AVG(return_5d) AS r5d
        FROM alert_evaluation
        """
    ).fetchone()
    if not row:
        return {"return_5m": None, "return_30m": None, "return_1d": None, "return_5d": None}
    return {"return_5m": row["r5m"], "return_30m": row["r30m"], "return_1d": row["r1d"], "return_5d": row["r5d"]}


def evaluation_summary(conn: sqlite3.Connection) -> dict:
    return {
        "precision_all": precision(conn),
        "precision_holdout": precision(conn, holdout=True),
        "precision_non_holdout": precision(conn, holdout=False),
        "false_positive_rate": false_positive_rate(conn),
        "average_latency_seconds": average_latency_seconds(conn),
        "average_return_by_horizon": average_return_by_horizon(conn),
    }
