"""Price, volume, and options-implied-volatility ingestion.

Price/volume: REAL, via Yahoo Finance's public chart endpoint
(`query1.finance.yahoo.com/v8/finance/chart/...`). No API key required.
We call this endpoint directly with `requests` rather than via the
`yfinance` package: yfinance's bundled HTTP client (curl_cffi, used to
impersonate a browser TLS fingerprint) does not play well with some
outbound proxies, while a plain `requests.get` against the same
endpoint works fine and needs no cookie/crumb dance for chart data.

Note Yahoo's intraday history is only available for a rolling window
(~60 days at 5m granularity), which is fine for computing 5m/1h/1d/5d
reactions to *recent* events but not for a long-horizon backtest -- see
README for the production swap-out recommendation (Polygon.io, IEX
Cloud, etc.).

Options-implied volatility: *** STUBBED / MOCKED ***. No free, reliable
IV data source was identified for the MVP. `iv_change` in
`market_reaction` is always populated by `MockOptionsIVProvider` until a
real vendor (CBOE DataShop, ORATS, Polygon.io options, Tradier, ...) is
wired up behind the `OptionsIVProvider` interface below.
"""
from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import requests

from market_intel.config import settings

logger = logging.getLogger(__name__)

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (market-intel MVP; +https://example.com)"}


class MarketDataIngestor:
    """Real price/volume via Yahoo's chart endpoint. Not a BaseIngestor:
    operates on a single ticker at a time and returns raw price-series
    rows rather than RawItems (price data isn't an "event" until scoring
    interprets it)."""

    is_mocked = False

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    def fetch_intraday(self, ticker: str, range_: str = "5d", interval: str = "5m") -> list[dict]:
        return self._fetch(ticker, range_, interval, granularity="intraday")

    def fetch_daily(self, ticker: str, range_: str = "6mo") -> list[dict]:
        return self._fetch(ticker, range_, "1d", granularity="daily")

    def _fetch(self, ticker: str, range_: str, interval: str, granularity: str) -> list[dict]:
        try:
            resp = self.session.get(
                YAHOO_CHART_URL.format(ticker=ticker),
                params={"range": range_, "interval": interval},
                headers=_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            payload = resp.json()
            result = payload["chart"]["result"][0]
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("Yahoo chart fetch failed for %s (%s/%s): %s", ticker, range_, interval, exc)
            return []

        timestamps = result.get("timestamp") or []
        quote = result.get("indicators", {}).get("quote", [{}])[0]
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []

        rows = []
        for i, ts in enumerate(timestamps):
            close = closes[i] if i < len(closes) else None
            if close is None:
                continue
            dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp_utc": dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "granularity": granularity,
                    "price": float(close),
                    "volume": float(volumes[i]) if i < len(volumes) and volumes[i] is not None else None,
                    "is_mocked": False,
                }
            )
        return rows


def persist_price_snapshots(conn, rows: list[dict]) -> None:
    for r in rows:
        conn.execute(
            """
            INSERT OR IGNORE INTO price_snapshots (ticker, timestamp_utc, granularity, price, volume, is_mocked)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                r["ticker"], r["timestamp_utc"], r.get("granularity", "daily"),
                r["price"], r.get("volume"), int(r.get("is_mocked", False)),
            ),
        )


class OptionsIVProvider(ABC):
    is_mocked = True

    @abstractmethod
    def get_iv_change(self, ticker: str, event_timestamp_utc: str) -> float | None:
        """Return the change in ATM implied volatility around the event
        (e.g. post-event IV minus pre-event IV), or None if unavailable."""
        raise NotImplementedError


class MockOptionsIVProvider(OptionsIVProvider):
    is_mocked = True

    def get_iv_change(self, ticker: str, event_timestamp_utc: str) -> float | None:
        seed = int(hashlib.sha256(f"{ticker}|{event_timestamp_utc}".encode()).hexdigest(), 16)
        iv_change = ((seed % 45) - 15) / 100  # -0.15 .. +0.30
        logger.info("[MOCK] iv_change for %s @ %s = %.3f", ticker, event_timestamp_utc, iv_change)
        return round(iv_change, 3)


def get_options_provider() -> OptionsIVProvider:
    if settings.options_is_mocked:
        return MockOptionsIVProvider()
    raise NotImplementedError(
        f"OPTIONS_PROVIDER={settings.options_provider!r} is not wired up yet. "
        "Implement OptionsIVProvider for it in ingestion/market_data.py."
    )
