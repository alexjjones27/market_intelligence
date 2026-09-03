"""News ingestion.

REAL when FINNHUB_API_KEY is set (Finnhub free-tier `/company-news`
endpoint: https://finnhub.io/docs/api/company-news). Chosen as the
default "one reliable news feed" per the build spec (free tier, no
credit card).

MOCK fallback when no key is configured: returns a small deterministic
set of synthetic articles per ticker so the rest of the pipeline
(dedup, scoring, alerts) is exercisable end-to-end without a key. Every
mocked item is tagged `is_mocked=True` and `source_tier="professional"`
so it is easy to filter out of anything that matters.

FLAG FOR WIRING: swap in a paid/broader feed (Benzinga, RavenPack, etc.)
by implementing another BaseIngestor with the same `fetch()` contract.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import requests

from market_intel.config import settings
from market_intel.ingestion.base import BaseIngestor
from market_intel.models.raw_item import RawItem

logger = logging.getLogger(__name__)

FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"


class NewsIngestor(BaseIngestor):
    source_tier = "professional"

    def __init__(self, lookback_days: int = 5, api_key: str | None = None, session: requests.Session | None = None):
        self.lookback_days = lookback_days
        self.api_key = api_key or settings.finnhub_api_key
        self.session = session or requests.Session()
        self.is_mocked = self.api_key is None
        if self.is_mocked:
            logger.warning(
                "FINNHUB_API_KEY not set -- NewsIngestor is running in MOCK mode. "
                "Set FINNHUB_API_KEY in .env for real news ingestion."
            )

    def fetch(self, tickers: list[str]) -> list[RawItem]:
        if self.is_mocked:
            return self._fetch_mock(tickers)
        return self._fetch_real(tickers)

    def _fetch_real(self, tickers: list[str]) -> list[RawItem]:
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=self.lookback_days)
        out: list[RawItem] = []
        for ticker in tickers:
            try:
                resp = self.session.get(
                    FINNHUB_NEWS_URL,
                    params={
                        "symbol": ticker,
                        "from": start.isoformat(),
                        "to": end.isoformat(),
                        "token": self.api_key,
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                articles = resp.json()
            except requests.RequestException as exc:
                logger.warning("Finnhub news fetch failed for %s: %s", ticker, exc)
                continue

            for art in articles:
                published_at = None
                if art.get("datetime"):
                    published_at = datetime.fromtimestamp(
                        art["datetime"], tz=timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                out.append(
                    RawItem(
                        source="Finnhub News",
                        source_tier="professional",
                        publisher=art.get("source"),
                        url=art.get("url"),
                        ticker_guess=ticker,
                        company_guess=None,
                        title=art.get("headline"),
                        body_text=art.get("summary"),
                        event_type_guess=None,  # left to processing/event_classification.py
                        published_at=published_at,
                        event_timestamp_guess=published_at,
                        retrieved_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        confirmed=False,  # news is unverified until corroborated by a primary source
                        is_mocked=False,
                    )
                )
        return out

    def _fetch_mock(self, tickers: list[str]) -> list[RawItem]:
        now = datetime.now(timezone.utc)
        out: list[RawItem] = []
        for ticker in tickers:
            published_at = now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            out.append(
                RawItem(
                    source="Mock News Feed",
                    source_tier="professional",
                    publisher="Mock Wire",
                    url=f"https://example.com/mock-news/{ticker.lower()}-earnings",
                    ticker_guess=ticker,
                    company_guess=None,
                    title=f"{ticker} beats on top and bottom line, raises guidance",
                    body_text=(
                        f"[MOCK DATA] {ticker} reported quarterly results ahead of "
                        "analyst expectations and raised full-year guidance."
                    ),
                    event_type_guess="earnings",
                    published_at=published_at,
                    event_timestamp_guess=published_at,
                    retrieved_at=published_at,
                    confirmed=False,
                    is_mocked=True,
                )
            )
        return out
