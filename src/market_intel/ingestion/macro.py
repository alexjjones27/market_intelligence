"""Macro release ingestion (CPI, core CPI, PCE, NFP, unemployment, retail
sales, GDP, jobless claims, Fed funds rate, ...).

REAL when FRED_API_KEY is set (https://fred.stlouisfed.org/docs/api/api_key.html,
free). Pulls from FRED's `series/observations` endpoint.

MOCK fallback when no key is configured: synthetic releases, flagged
`is_mocked=True` on every row.

Known limitations (flagged, not silently papered over):
  * FRED does not carry a "consensus"/expected value -- that requires a
    separate (typically paid) surveys feed. `consensus` is always None
    until one is wired up; scoring code must handle that.
  * FRED's default endpoint returns the latest-vintage value, not the
    as-first-published (ALFRED) vintage. True point-in-time ALFRED
    querying (`realtime_start`/`realtime_end`) is NOT implemented in this
    MVP. Instead we approximate point-in-time discipline at the
    database layer: the first time we observe a value for a given
    (series, period) we store it and mark `is_first_release=1`; if a
    later fetch for the same period returns a different value, that is
    stored as a NEW row with `is_first_release=0` -- so our own
    revision history is preserved even though it starts from whenever
    we started polling, not from the true initial release. See
    db/schema.sql: macro_releases.
  * ISM Manufacturing/Services PMI is licensed by ISM and not on FRED.
    It is listed in MACRO_SERIES with a `None` FRED id and always comes
    back mocked until a paid source is wired up.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

import requests

from market_intel.config import settings

logger = logging.getLogger(__name__)

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

# release_name -> FRED series id (None = not available on FRED; mocked only)
MACRO_SERIES: dict[str, str | None] = {
    "CPI": "CPIAUCSL",
    "CORE_CPI": "CPILFESL",
    "PCE": "PCEPI",
    "CORE_PCE": "PCEPILFE",
    "UNEMPLOYMENT_RATE": "UNRATE",
    "NONFARM_PAYROLLS": "PAYEMS",
    "RETAIL_SALES": "RSAFS",
    "GDP": "GDPC1",
    "JOBLESS_CLAIMS": "ICSA",
    "FED_FUNDS_RATE": "FEDFUNDS",
    "ISM_MANUFACTURING": None,  # not on FRED; licensed data, mocked only
}


class MacroIngestor:
    """Not a BaseIngestor: macro releases are ticker-agnostic, so the
    per-ticker `fetch(tickers)` contract doesn't fit. Call `fetch_releases`
    directly."""

    def __init__(self, api_key: str | None = None, session: requests.Session | None = None):
        self.api_key = api_key or settings.fred_api_key
        self.session = session or requests.Session()
        self.is_mocked = self.api_key is None
        if self.is_mocked:
            logger.warning(
                "FRED_API_KEY not set -- MacroIngestor is running in MOCK mode. "
                "Set FRED_API_KEY in .env for real macro data."
            )

    def fetch_releases(self, release_names: list[str] | None = None) -> list[dict]:
        names = release_names or list(MACRO_SERIES.keys())
        out: list[dict] = []
        for name in names:
            series_id = MACRO_SERIES.get(name)
            if self.is_mocked or series_id is None:
                out.append(self._mock_release(name, series_id))
                continue
            try:
                out.append(self._fetch_one(name, series_id))
            except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
                logger.warning("FRED fetch failed for %s (%s): %s -- falling back to mock", name, series_id, exc)
                out.append(self._mock_release(name, series_id))
        return out

    def _fetch_one(self, name: str, series_id: str) -> dict:
        resp = self.session.get(
            FRED_OBSERVATIONS_URL,
            params={
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 2,
            },
            timeout=15,
        )
        resp.raise_for_status()
        obs = resp.json()["observations"]
        latest, prior = obs[0], obs[1] if len(obs) > 1 else {"value": None}

        return {
            "series_id": series_id,
            "release_name": name,
            "period": latest["date"],
            "value": float(latest["value"]),
            "prior_period_value": float(prior["value"]) if prior.get("value") not in (None, ".") else None,
            "consensus": None,  # not available from FRED; see module docstring
            "published_at": f"{latest['date']}T00:00:00.000000Z",
            "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "is_mocked": False,
            "source": "FRED",
        }

    @staticmethod
    def _mock_release(name: str, series_id: str | None) -> dict:
        seed = int(hashlib.sha256(name.encode()).hexdigest(), 16)
        base = 100 + (seed % 900) / 10
        now = datetime.now(timezone.utc)
        period = now.strftime("%Y-%m")
        logger.info("[MOCK] macro release %s (no FRED series or no key)", name)
        return {
            "series_id": series_id or f"MOCK_{name}",
            "release_name": name,
            "period": period,
            "value": round(base, 2),
            "prior_period_value": round(base * 0.995, 2),
            "consensus": round(base * 1.002, 2),
            "published_at": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "retrieved_at": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "is_mocked": True,
            "source": "mock",
        }


def persist_releases(conn, releases: list[dict]) -> None:
    """Insert-only, point-in-time-preserving persistence: never overwrites
    a previously stored value for (series_id, period). A changed value for
    an already-seen period is stored as an additional revision row."""
    for r in releases:
        existing = conn.execute(
            "SELECT value FROM macro_releases WHERE series_id = ? AND period = ? "
            "ORDER BY published_at DESC, id DESC LIMIT 1",
            (r["series_id"], r["period"]),
        ).fetchone()

        if existing is None:
            is_first = 1
        elif float(existing["value"]) == float(r["value"]):
            continue  # unchanged, nothing to record
        else:
            is_first = 0

        conn.execute(
            """
            INSERT INTO macro_releases
                (series_id, release_name, period, value, is_first_release,
                 consensus, prior_period_value, published_at, retrieved_at, is_mocked, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["series_id"], r["release_name"], r["period"], r["value"], is_first,
                r.get("consensus"), r.get("prior_period_value"), r["published_at"],
                r["retrieved_at"], int(r.get("is_mocked", False)), r.get("source", "FRED"),
            ),
        )
