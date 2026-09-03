"""UTC normalization + exchange-local time helpers.

Every event timestamp in the system is stored UTC-normalized; the
exchange-local time is carried alongside for human-readable output (e.g.
"was this released before/after the bell") but is never used as the
source of truth for ordering or windowing.
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

US_EQUITY_TZ = ZoneInfo("America/New_York")
ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def parse_utc(ts: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp string into an aware datetime."""
    ts = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(ISO_FMT)


def to_exchange_local_iso(dt: datetime, tz: ZoneInfo = US_EQUITY_TZ) -> str:
    return dt.astimezone(tz).isoformat()


def normalize(ts: str, tz: ZoneInfo = US_EQUITY_TZ) -> tuple[str, str]:
    """Given any parseable ISO timestamp, return (utc_iso, exchange_local_iso)."""
    dt = parse_utc(ts)
    return to_utc_iso(dt), to_exchange_local_iso(dt, tz)


def is_regular_session(dt_local: datetime) -> bool:
    """True if a US-equity-local datetime falls within 09:30-16:00 ET on a weekday."""
    if dt_local.weekday() >= 5:
        return False
    minutes = dt_local.hour * 60 + dt_local.minute
    return 9 * 60 + 30 <= minutes <= 16 * 60
