"""Canonical event schema.

Mirrors the JSON object in the spec exactly, as three independent,
never-mixed sections:

  * ``facts``          -- what happened, extracted verbatim from source text.
                           Never written to by an LLM's own inference.
  * ``interpretation``  -- our derived scores (surprise, confidence, etc).
  * ``market_reaction``  -- observed price/volume/IV response, populated
                             asynchronously as data arrives after the event.

``evidence`` is a list because one event (e.g. an earnings release) is
typically corroborated by multiple source items (press release, 8-K,
transcript, several news articles) -- see processing/dedup.py.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class SourceTier(str, Enum):
    PRIMARY = "primary"          # SEC EDGAR, company IR
    PROFESSIONAL = "professional"  # news feed, earnings-call transcripts
    SECONDARY = "secondary"       # context-only, unused in MVP scoring gates


class EventType(str, Enum):
    EARNINGS = "earnings"
    GUIDANCE_CHANGE = "guidance_change"
    M_AND_A = "m_and_a"
    EXEC_CHANGE = "exec_change"
    LEGAL_REGULATORY = "legal_regulatory"
    CAPITAL_ALLOCATION = "capital_allocation"
    MACRO = "macro"
    OTHER = "other"


class Evidence(BaseModel):
    source: str
    url: Optional[str] = None
    publisher: Optional[str] = None
    published_at: Optional[str] = None
    retrieved_at: str
    confirmed: bool = False
    raw_item_id: Optional[str] = None


class Facts(BaseModel):
    """What was reported. Extend freely -- unknown keys pass through so
    event-type-specific facts (deal value, exec name, fine amount, ...)
    don't require a schema migration."""

    model_config = ConfigDict(extra="allow")

    eps_actual: Optional[float] = None
    eps_consensus: Optional[float] = None
    revenue_actual: Optional[float] = None
    revenue_consensus: Optional[float] = None
    guidance_direction: Optional[str] = None  # raised | lowered | maintained | withdrawn


class Exposure(BaseModel):
    sectors: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    commodities: list[str] = Field(default_factory=list)
    peers: list[str] = Field(default_factory=list)
    sector_etfs: list[str] = Field(default_factory=list)


class Interpretation(BaseModel):
    """Derived scores only. Nothing here is a reported fact."""

    surprise_score: Optional[float] = None
    novelty_score: Optional[float] = None
    market_sensitivity_score: Optional[float] = None
    confirmation_score: Optional[float] = None
    forward_earnings_effect: Optional[str] = None  # positive | negative | neutral
    macro_sensitivity: Optional[str] = None  # low | medium | high
    confidence: Optional[float] = None
    verification_flag: bool = False


class MarketReaction(BaseModel):
    """Observed, not inferred. Populated asynchronously; all fields start
    null until the corresponding price/volume/IV data has arrived."""

    return_5m: Optional[float] = None
    return_1h: Optional[float] = None
    return_1d: Optional[float] = None
    return_5d: Optional[float] = None
    volume_zscore: Optional[float] = None
    iv_change: Optional[float] = None


class Event(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    company: str
    ticker: str
    timestamp_utc: str
    exchange_local_time: Optional[str] = None
    source_tier: SourceTier

    evidence: list[Evidence] = Field(default_factory=list)
    facts: Facts = Field(default_factory=Facts)
    exposure: Exposure = Field(default_factory=Exposure)
    interpretation: Interpretation = Field(default_factory=Interpretation)
    market_reaction: MarketReaction = Field(default_factory=MarketReaction)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    def to_row(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "company": self.company,
            "ticker": self.ticker,
            "timestamp_utc": self.timestamp_utc,
            "exchange_local_time": self.exchange_local_time,
            "source_tier": self.source_tier.value,
            "facts": self.facts.model_dump(),
            "exposure": self.exposure.model_dump(),
            "interpretation": self.interpretation.model_dump(),
            "market_reaction": self.market_reaction.model_dump(),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any], evidence: list[dict[str, Any]] | None = None) -> "Event":
        return cls(
            event_id=row["event_id"],
            event_type=row["event_type"],
            company=row["company"],
            ticker=row["ticker"],
            timestamp_utc=row["timestamp_utc"],
            exchange_local_time=row.get("exchange_local_time"),
            source_tier=row["source_tier"],
            facts=Facts(**(row.get("facts") or {})),
            exposure=Exposure(**(row.get("exposure") or {})),
            interpretation=Interpretation(**(row.get("interpretation") or {})),
            market_reaction=MarketReaction(**(row.get("market_reaction") or {})),
            evidence=[Evidence(**e) for e in (evidence or [])],
        )
