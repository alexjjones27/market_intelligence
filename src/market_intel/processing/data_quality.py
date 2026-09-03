"""Data-quality gate, run BEFORE an event is allowed to compete for an
alert -- not a post-hoc audit. Every check here answers "is this number
even trustworthy," which is a different question from "is this
surprising" or "did the market react."

Two severities:
  * warning -- known limitation or soft anomaly; the event stays
    eligible but the issue is surfaced (in `build_risk_note` and the
    quarantine view) so nobody mistakes an unverified number for a
    solid one.
  * reject  -- the event is quarantined from immediate alerts
    (`blocking=True`, checked in alert_engine.passes_immediate_alert_gate).
    It still exists, still shows up in the morning brief and the
    quarantine CLI view -- rejection means "don't trust this enough to
    page someone," not "delete it."
"""
from __future__ import annotations

import sqlite3

from market_intel.models.event import DataQuality, Event
from market_intel.processing.timestamps import parse_utc

# A revenue/EPS actual more than this many times its consensus (or less
# than the reciprocal) almost always means a units/scale mismatch
# (millions vs. dollars, EPS vs. net income, ...) rather than a genuine
# 100x surprise.
SCALE_MISMATCH_RATIO = 10.0

# How far apart an event's evidence timestamps are allowed to span before
# we no longer trust the cluster as "one disclosure." Generous on
# purpose -- this is a sanity backstop, not the primary defense (that's
# dedup.py's clustering window itself); see dedup.MAX_NEAR_DUPLICATE_GAP.
MAX_EVIDENCE_SPAN_DAYS = 21


def _scale_mismatch(actual: float | None, consensus: float | None) -> bool:
    if actual is None or consensus is None or consensus == 0 or actual == 0:
        return False
    ratio = abs(actual / consensus)
    return ratio > SCALE_MISMATCH_RATIO or ratio < (1 / SCALE_MISMATCH_RATIO)


def check_reported_value_scale(event: Event) -> str | None:
    f = event.facts
    if _scale_mismatch(f.eps_actual, f.eps_consensus) or _scale_mismatch(f.revenue_actual, f.revenue_consensus):
        return "reported_value_scale_suspicious"
    return None


def check_duplicate_evidence(event: Event) -> str | None:
    seen = set()
    for ev in event.evidence:
        key = ev.url or (ev.source, ev.published_at)
        if key in seen:
            return "possible_duplicate_filing"
        seen.add(key)
    return None


def check_event_span(event: Event) -> str | None:
    timestamps = [ev.published_at for ev in event.evidence if ev.published_at]
    if len(timestamps) < 2:
        return None
    parsed = [parse_utc(t) for t in timestamps]
    span_days = (max(parsed) - min(parsed)).days
    if span_days > MAX_EVIDENCE_SPAN_DAYS:
        return "event_date_inconsistent"
    return None


def check_consensus_point_in_time(conn: sqlite3.Connection, event: Event) -> str | None:
    """The consensus figures attached to an earnings event must have
    been known BEFORE the event, or the surprise score leaks future
    information into the past. This MVP's earnings provider is fully
    mocked (see ingestion/earnings.py) and always captures "consensus"
    at pipeline-run time -- so for any event whose real-world disclosure
    predates this run, that consensus number was NOT actually available
    at event time. This is flagged, not hidden, on every such event
    until a real point-in-time consensus feed is wired up."""
    if event.event_type.value != "earnings" or event.facts.eps_consensus is None:
        return None
    row = conn.execute(
        "SELECT first_captured_at FROM earnings_consensus WHERE ticker = ? "
        "ORDER BY first_captured_at DESC LIMIT 1",
        (event.ticker,),
    ).fetchone()
    if not row:
        return None
    try:
        if parse_utc(row["first_captured_at"]) > parse_utc(event.timestamp_utc):
            return "consensus_not_point_in_time"
    except ValueError:
        pass
    return None


def check_earnings_data_sufficiency(event: Event) -> str | None:
    if event.event_type.value != "earnings":
        return None
    if event.facts.eps_actual is None and event.facts.revenue_actual is None:
        return "insufficient_earnings_data"
    return None


# issue -> severity. Anything not listed defaults to "warning".
REJECT_ISSUES = {"reported_value_scale_suspicious", "possible_duplicate_filing", "event_date_inconsistent"}


def evaluate_data_quality(conn: sqlite3.Connection, event: Event) -> DataQuality:
    issues = [
        issue
        for issue in (
            check_reported_value_scale(event),
            check_duplicate_evidence(event),
            check_event_span(event),
            check_consensus_point_in_time(conn, event),
            check_earnings_data_sufficiency(event),
        )
        if issue is not None
    ]

    if not issues:
        return DataQuality(status="pass", issues=[], blocking=False)

    blocking = any(issue in REJECT_ISSUES for issue in issues)
    status = "reject" if blocking else "warning"
    return DataQuality(status=status, issues=issues, blocking=blocking)
