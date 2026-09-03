"""SEC EDGAR ingestion. REAL data source -- no API key required, but SEC
requires a descriptive User-Agent on every request (see
config.settings.sec_user_agent / .env.example SEC_EDGAR_USER_AGENT).

Pulls each watchlist ticker's recent 8-K / 10-Q / 10-K filings via the
public submissions API:
    https://data.sec.gov/submissions/CIK{cik10}.json

For 8-Ks, the filing's disclosed "Item" numbers (e.g. "5.02", "2.02") are
used to make a first-pass event-type guess -- this is SEC-specific domain
knowledge, so it lives here rather than in the generic classifier, which
still gets the final say once evidence is clustered.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from market_intel.config import settings
from market_intel.ingestion.base import BaseIngestor
from market_intel.models.raw_item import RawItem

logger = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{doc}"

RELEVANT_FORMS = {"8-K", "8-K/A", "10-Q", "10-Q/A", "10-K", "10-K/A"}

# 8-K Item number -> best-guess event type. Deliberately coarse; the
# classifier in processing/event_classification.py reconciles this with
# text-based signals once evidence is clustered.
ITEM_TO_EVENT_TYPE = {
    "1.01": "capital_allocation",
    "1.02": "other",
    "1.03": "legal_regulatory",
    "2.01": "m_and_a",
    "2.02": "earnings",
    "2.05": "capital_allocation",
    "2.06": "capital_allocation",
    "3.01": "legal_regulatory",
    "3.02": "capital_allocation",
    "4.01": "legal_regulatory",
    "4.02": "legal_regulatory",
    "5.01": "m_and_a",
    "5.02": "exec_change",
    "5.03": "other",
    "5.07": "other",
    "7.01": "guidance_change",
    "8.01": "other",
}


class SecEdgarIngestor(BaseIngestor):
    source_tier = "primary"
    is_mocked = False

    def __init__(
        self,
        lookback_days: int = 10,
        watchlist_path: str | None = None,
        session: requests.Session | None = None,
        request_delay_s: float = 0.15,
    ):
        self.lookback_days = lookback_days
        self.watchlist_path = watchlist_path or settings.watchlist_path
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": settings.sec_user_agent})
        self.request_delay_s = request_delay_s

    def _ticker_to_cik(self) -> dict[str, str]:
        data = json.loads(Path(self.watchlist_path).read_text())
        return {
            e["ticker"]: e["cik"]
            for e in data.get("tickers", [])
            if e.get("cik")
        }

    def fetch(self, tickers: list[str]) -> list[RawItem]:
        ticker_to_cik = self._ticker_to_cik()
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        items: list[RawItem] = []

        for ticker in tickers:
            cik = ticker_to_cik.get(ticker)
            if not cik:
                logger.warning("No CIK on file for %s; skipping SEC EDGAR fetch", ticker)
                continue
            try:
                items.extend(self._fetch_one(ticker, cik, cutoff))
            except requests.RequestException as exc:
                logger.warning("SEC EDGAR fetch failed for %s: %s", ticker, exc)
            time.sleep(self.request_delay_s)

        return items

    def _fetch_one(self, ticker: str, cik: str, cutoff: datetime) -> list[RawItem]:
        cik10 = cik.zfill(10)
        url = SUBMISSIONS_URL.format(cik10=cik10)
        resp = self.session.get(url, timeout=15)
        resp.raise_for_status()
        payload = resp.json()

        company_name = payload.get("name", ticker)
        recent = payload.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        items_field = recent.get("items", [""] * len(forms))

        out: list[RawItem] = []
        for i, form in enumerate(forms):
            if form not in RELEVANT_FORMS:
                continue
            filing_date_str = filing_dates[i]
            filing_dt = datetime.strptime(filing_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if filing_dt < cutoff:
                continue

            accession_nodash = accession_numbers[i].replace("-", "")
            doc = primary_docs[i] if i < len(primary_docs) else ""
            cik_int = str(int(cik))
            filing_url = ARCHIVES_URL.format(cik_int=cik_int, accession_nodash=accession_nodash, doc=doc)

            item_codes = [c.strip() for c in (items_field[i] if i < len(items_field) else "").split(",") if c.strip()]
            event_type_guess = self._guess_event_type(form, item_codes)

            report_date = report_dates[i] if i < len(report_dates) and report_dates[i] else filing_date_str
            event_ts = f"{report_date}T00:00:00.000000Z"
            published_ts = f"{filing_date_str}T00:00:00.000000Z"

            out.append(
                RawItem(
                    source=f"SEC {form}",
                    source_tier="primary",
                    publisher="SEC EDGAR",
                    url=filing_url,
                    ticker_guess=ticker,
                    company_guess=company_name,
                    title=f"{company_name} {form} filing"
                    + (f" (items {', '.join(item_codes)})" if item_codes else ""),
                    body_text=None,  # MVP: metadata-only; full-text fetch is a TODO (see README)
                    event_type_guess=event_type_guess,
                    published_at=published_ts,
                    event_timestamp_guess=event_ts,
                    retrieved_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    confirmed=True,  # primary source, filed directly with the SEC
                    is_mocked=False,
                )
            )
        return out

    @staticmethod
    def _guess_event_type(form: str, item_codes: list[str]) -> str:
        if form.startswith("10-Q") or form.startswith("10-K"):
            return "earnings"
        for code in item_codes:
            if code in ITEM_TO_EVENT_TYPE:
                return ITEM_TO_EVENT_TYPE[code]
        return "other"
