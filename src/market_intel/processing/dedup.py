"""Deduplication / clustering.

A single real-world event (e.g. an earnings release) typically shows up
as several raw items: an IR press release, an 8-K, an earnings-call
transcript, and multiple news articles. This module collapses such a
group into ONE Event with MULTIPLE Evidence records -- never multiple
events for the same underlying happening.

Clustering key: (resolved ticker, event type, time window). A secondary
near-duplicate text check (stdlib `difflib`, no embeddings needed for
MVP scale) lets textually-identical items join a cluster even if they
land just outside the window, e.g. a wire story re-published a few hours
late by a secondary outlet.
"""
from __future__ import annotations

import difflib
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from market_intel.models.event import Event, EventType, Evidence, SourceTier
from market_intel.models.raw_item import RawItem
from market_intel.processing.timestamps import normalize, parse_utc

DEFAULT_TIME_WINDOW = timedelta(hours=48)
EARNINGS_TIME_WINDOW = timedelta(days=5)  # 8-K + 10-Q for the same release often land days apart

TIER_RANK = {"primary": 3, "professional": 2, "secondary": 1}


@dataclass
class Cluster:
    ticker: str
    event_type: str
    items: list[RawItem] = field(default_factory=list)


def _timestamp_of(item: RawItem) -> datetime:
    ts = item.event_timestamp_guess or item.published_at
    if ts:
        try:
            return parse_utc(ts)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def near_duplicate(a: RawItem, b: RawItem, threshold: float = 0.6) -> bool:
    """Title-similarity based near-dup check. Falls back to False (not a
    dup) whenever either title is missing -- we never want an empty
    string to spuriously match another empty string."""
    ta, tb = (a.title or "").strip().lower(), (b.title or "").strip().lower()
    if not ta or not tb:
        return False
    return difflib.SequenceMatcher(None, ta, tb).ratio() >= threshold


def cluster_raw_items(
    items: list[RawItem],
    time_window: timedelta = DEFAULT_TIME_WINDOW,
    earnings_time_window: timedelta = EARNINGS_TIME_WINDOW,
) -> list[Cluster]:
    """Group raw items into clusters. Items without a resolved ticker are
    dropped (caller should have run entity resolution first and either
    filled ticker_guess or excluded the item)."""
    buckets: dict[tuple[str, str], list[RawItem]] = defaultdict(list)
    for item in items:
        if not item.ticker_guess:
            continue
        key = (item.ticker_guess, item.event_type_guess or "other")
        buckets[key].append(item)

    clusters: list[Cluster] = []
    for (ticker, event_type), group in buckets.items():
        window = earnings_time_window if event_type == "earnings" else time_window
        group_sorted = sorted(group, key=_timestamp_of)

        open_clusters: list[Cluster] = []
        for item in group_sorted:
            item_ts = _timestamp_of(item)
            placed = False
            for cluster in open_clusters:
                anchor_ts = _timestamp_of(cluster.items[0])
                within_window = (item_ts - anchor_ts) <= window
                is_dup = any(near_duplicate(item, existing) for existing in cluster.items)
                if within_window or is_dup:
                    cluster.items.append(item)
                    placed = True
                    break
            if not placed:
                open_clusters.append(Cluster(ticker=ticker, event_type=event_type, items=[item]))
        clusters.extend(open_clusters)

    return clusters


def _best_source_tier(items: list[RawItem]) -> str:
    return max(items, key=lambda it: TIER_RANK.get(it.source_tier, 0)).source_tier


def _best_company_name(items: list[RawItem]) -> str:
    for item in items:
        if item.company_guess:
            return item.company_guess
    return items[0].ticker_guess or "UNKNOWN"


def cluster_to_event(cluster: Cluster) -> Event:
    earliest_item = min(cluster.items, key=_timestamp_of)
    event_ts_raw = earliest_item.event_timestamp_guess or earliest_item.published_at
    if event_ts_raw:
        timestamp_utc, exchange_local_time = normalize(event_ts_raw)
    else:
        timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        exchange_local_time = None

    try:
        event_type = EventType(cluster.event_type)
    except ValueError:
        event_type = EventType.OTHER

    evidence = [
        Evidence(
            source=it.source,
            url=it.url,
            publisher=it.publisher,
            published_at=it.published_at,
            retrieved_at=it.retrieved_at,
            confirmed=it.confirmed,
            raw_item_id=it.raw_item_id,
        )
        for it in cluster.items
    ]

    return Event(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        company=_best_company_name(cluster.items),
        ticker=cluster.ticker,
        timestamp_utc=timestamp_utc,
        exchange_local_time=exchange_local_time,
        source_tier=SourceTier(_best_source_tier(cluster.items)),
        evidence=evidence,
    )


def cluster_and_build_events(items: list[RawItem]) -> list[Event]:
    return [cluster_to_event(c) for c in cluster_raw_items(items)]
