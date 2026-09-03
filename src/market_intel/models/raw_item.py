"""Pre-clustering ingested item. Every ingestion module (SEC, news,
earnings, macro) emits these; processing/dedup.py collapses groups of
them into a single Event with multiple Evidence records.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class RawItem(BaseModel):
    raw_item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str                      # "SEC 8-K", "SEC 10-Q", "Finnhub News", "IR Release", ...
    source_tier: str                 # primary | professional | secondary
    publisher: Optional[str] = None
    url: Optional[str] = None
    ticker_guess: Optional[str] = None
    company_guess: Optional[str] = None
    title: Optional[str] = None
    body_text: Optional[str] = None
    event_type_guess: Optional[str] = None
    published_at: Optional[str] = None          # UTC ISO8601
    event_timestamp_guess: Optional[str] = None  # UTC ISO8601 -- when this became PUBLIC, not
                                                  # necessarily when the underlying fact pattern
                                                  # occurred (see fiscal_period_end below).
    fiscal_period_end: Optional[str] = None      # ISO date, e.g. "2026-07-26". For 10-Q/10-K/
                                                  # earnings 8-Ks: the fiscal quarter/year this
                                                  # filing REPORTS ON, distinct from when it was
                                                  # disclosed. Deliberately NOT persisted to the
                                                  # raw_items table (in-memory only, consumed by
                                                  # dedup.cluster_to_event into Event.facts) --
                                                  # it's derived metadata, not a source attribute.
    exchange_local_time: Optional[str] = None
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    )
    confirmed: bool = False
    is_mocked: bool = False

    @property
    def content_hash(self) -> str:
        basis = f"{self.ticker_guess}|{self.event_type_guess}|{(self.title or '').strip().lower()}"
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()

    def stabilize_id(self) -> None:
        """Recompute raw_item_id deterministically from (ticker, source,
        url) once the ticker is resolved. Must be called AFTER entity
        resolution (processing/entity_resolution.py) fills in
        ticker_guess, and BEFORE persistence/clustering -- otherwise a
        re-fetch of the same underlying item on a later pipeline run
        gets a fresh random id, the raw_items UNIQUE constraint silently
        ignores the insert (see db/repository.py:insert_raw_item), and
        any Evidence built from it points at a raw_item_id that was
        never actually persisted -- a foreign-key violation. No-ops
        when there's no url to key off (can't dedupe those anyway; the
        constructor's random uuid4 stands)."""
        if self.ticker_guess and self.url:
            basis = f"{self.ticker_guess}|{self.source}|{self.url}"
            self.raw_item_id = str(uuid.uuid5(uuid.NAMESPACE_URL, basis))

    def to_row(self) -> dict[str, Any]:
        return {
            "raw_item_id": self.raw_item_id,
            "source": self.source,
            "source_tier": self.source_tier,
            "publisher": self.publisher,
            "url": self.url,
            "ticker_guess": self.ticker_guess,
            "company_guess": self.company_guess,
            "title": self.title,
            "body_text": self.body_text,
            "event_type_guess": self.event_type_guess,
            "published_at": self.published_at,
            "event_timestamp_guess": self.event_timestamp_guess,
            "exchange_local_time": self.exchange_local_time,
            "retrieved_at": self.retrieved_at,
            "confirmed": int(self.confirmed),
            "content_hash": self.content_hash,
            "is_mocked": int(self.is_mocked),
        }
