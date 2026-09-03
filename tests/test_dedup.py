"""Dedup/clustering tests.

Core requirement from the build spec: a press release + an 8-K + two
news articles about the same earnings release must collapse into ONE
event with FOUR evidence records, not four events.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from market_intel.models.raw_item import RawItem
from market_intel.processing.dedup import cluster_and_build_events, cluster_raw_items, near_duplicate


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _make_earnings_cluster_items() -> list[RawItem]:
    base = datetime(2026, 7, 30, 20, 5, tzinfo=timezone.utc)
    return [
        RawItem(
            source="IR Press Release",
            source_tier="primary",
            publisher="Example Corp IR",
            url="https://ir.example.com/q3-2026-results",
            ticker_guess="EXM",
            company_guess="Example Corp",
            title="Example Corp Reports Third Quarter 2026 Results",
            body_text="Example Corp today reported results for its third quarter.",
            event_type_guess="earnings",
            published_at=_iso(base),
            event_timestamp_guess=_iso(base),
            confirmed=True,
        ),
        RawItem(
            source="SEC 8-K",
            source_tier="primary",
            publisher="SEC EDGAR",
            url="https://www.sec.gov/Archives/edgar/data/0/000000/exm-8k.htm",
            ticker_guess="EXM",
            company_guess="Example Corp",
            title="Example Corp 8-K filing (items 2.02, 9.01)",
            event_type_guess="earnings",
            published_at=_iso(base + timedelta(minutes=10)),
            event_timestamp_guess=_iso(base + timedelta(minutes=10)),
            confirmed=True,
        ),
        RawItem(
            source="Finnhub News",
            source_tier="professional",
            publisher="Reuters",
            url="https://news.example.com/exm-beats-estimates",
            ticker_guess="EXM",
            company_guess=None,
            title="Example Corp beats on top and bottom line",
            event_type_guess="earnings",
            published_at=_iso(base + timedelta(minutes=25)),
            event_timestamp_guess=_iso(base + timedelta(minutes=25)),
            confirmed=False,
        ),
        RawItem(
            source="Finnhub News",
            source_tier="professional",
            publisher="Bloomberg",
            url="https://news.example.com/exm-raises-guidance",
            ticker_guess="EXM",
            company_guess=None,
            title="Example Corp raises full-year guidance after strong quarter",
            event_type_guess="earnings",
            published_at=_iso(base + timedelta(hours=1)),
            event_timestamp_guess=_iso(base + timedelta(hours=1)),
            confirmed=False,
        ),
    ]


def test_press_release_8k_and_two_articles_collapse_into_one_event():
    items = _make_earnings_cluster_items()

    clusters = cluster_raw_items(items)

    assert len(clusters) == 1
    assert len(clusters[0].items) == 4

    events = cluster_and_build_events(items)
    assert len(events) == 1

    event = events[0]
    assert event.ticker == "EXM"
    assert event.event_type.value == "earnings"
    assert len(event.evidence) == 4
    assert event.source_tier.value == "primary"  # best tier among evidence wins
    # earliest item's timestamp becomes the event's canonical timestamp
    assert event.timestamp_utc == _iso(datetime(2026, 7, 30, 20, 5, tzinfo=timezone.utc))


def test_different_tickers_never_cluster_together():
    items = _make_earnings_cluster_items()
    other_ticker_item = items[0].model_copy(update={"ticker_guess": "OTHR", "raw_item_id": "different"})
    items.append(other_ticker_item)

    clusters = cluster_raw_items(items)

    assert len(clusters) == 2
    ticker_sets = {c.ticker for c in clusters}
    assert ticker_sets == {"EXM", "OTHR"}


def test_items_outside_window_and_not_near_duplicate_form_separate_clusters():
    base = datetime(2026, 7, 30, 20, 5, tzinfo=timezone.utc)
    early = RawItem(
        source="Finnhub News",
        source_tier="professional",
        ticker_guess="EXM",
        title="Example Corp announces new share buyback program",
        event_type_guess="capital_allocation",
        published_at=_iso(base),
        event_timestamp_guess=_iso(base),
    )
    late_unrelated = RawItem(
        source="Finnhub News",
        source_tier="professional",
        ticker_guess="EXM",
        title="Example Corp CFO to present at industry conference",
        event_type_guess="capital_allocation",
        published_at=_iso(base + timedelta(days=10)),
        event_timestamp_guess=_iso(base + timedelta(days=10)),
    )

    clusters = cluster_raw_items([early, late_unrelated])

    assert len(clusters) == 2


def test_near_duplicate_detects_similar_titles():
    a = RawItem(source="A", source_tier="professional", title="Example Corp beats estimates, raises guidance")
    b = RawItem(source="B", source_tier="professional", title="Example Corp beats estimates and raises guidance")
    c = RawItem(source="C", source_tier="professional", title="Completely unrelated headline about widgets")

    assert near_duplicate(a, b) is True
    assert near_duplicate(a, c) is False


def test_items_without_resolved_ticker_are_dropped():
    items = _make_earnings_cluster_items()
    items[0].ticker_guess = None

    clusters = cluster_raw_items(items)

    total_items = sum(len(c.items) for c in clusters)
    assert total_items == 3
