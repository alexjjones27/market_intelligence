"""End-to-end pipeline test on fully mocked data -- no network calls.

Patches the network-touching ingestors (SEC EDGAR, market data) with
deterministic synthetic responses and runs the real pipeline (dedup,
scoring, alerting, persistence) against them, per deliverable #6:
"a CLI or simple script that runs the pipeline end-to-end on 3-5
watchlist tickers and prints a ranked alert list."
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from market_intel.db.database import connect
from market_intel.models.raw_item import RawItem
from market_intel.ingestion import market_data as market_data_module
from market_intel.ingestion import sec_edgar as sec_edgar_module
from market_intel import pipeline as pipeline_module

ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"
EVENT_TIME = datetime(2026, 8, 30, 14, 0, tzinfo=timezone.utc)


def _fake_sec_fetch(self, tickers):
    items = []
    for ticker in tickers:
        items.append(
            RawItem(
                source="SEC 8-K",
                source_tier="primary",
                publisher="SEC EDGAR",
                url=f"https://www.sec.gov/Archives/edgar/fake/{ticker.lower()}-8k.htm",
                ticker_guess=ticker,
                company_guess=None,
                title=f"{ticker} 8-K filing (items 2.02, 9.01)",
                event_type_guess="earnings",
                published_at=EVENT_TIME.strftime(ISO_FMT),
                event_timestamp_guess=EVENT_TIME.strftime(ISO_FMT),
                retrieved_at=datetime.now(timezone.utc).strftime(ISO_FMT),
                confirmed=True,
                is_mocked=True,
            )
        )
    return items


def _fake_fetch_daily(self, ticker, range_="6mo"):
    base = EVENT_TIME - timedelta(days=40)
    rows = []
    price = 100.0
    for i in range(45):
        ts = base + timedelta(days=i)
        price *= 1.001
        rows.append(
            {
                "ticker": ticker,
                "timestamp_utc": ts.strftime(ISO_FMT),
                "granularity": "daily",
                "price": round(price, 2),
                "volume": 1_000_000 + i * 1_000,
                "is_mocked": True,
            }
        )
    return rows


def _fake_fetch_intraday(self, ticker, range_="5d", interval="5m"):
    base = EVENT_TIME - timedelta(hours=2)
    rows = []
    price = 140.0
    for i in range(120):
        ts = base + timedelta(minutes=5 * i)
        price *= 1.0007  # steady drift so 5m/1h/1d horizons all show a positive move
        rows.append(
            {
                "ticker": ticker,
                "timestamp_utc": ts.strftime(ISO_FMT),
                "granularity": "intraday",
                "price": round(price, 2),
                "volume": 500_000,
                "is_mocked": True,
            }
        )
    return rows


@pytest.fixture(autouse=True)
def _patch_network(monkeypatch):
    monkeypatch.setattr(sec_edgar_module.SecEdgarIngestor, "fetch", _fake_sec_fetch)
    monkeypatch.setattr(market_data_module.MarketDataIngestor, "fetch_daily", _fake_fetch_daily)
    monkeypatch.setattr(market_data_module.MarketDataIngestor, "fetch_intraday", _fake_fetch_intraday)
    monkeypatch.setattr(pipeline_module.NewsIngestor, "fetch", lambda self, tickers: [])


def test_pipeline_runs_end_to_end_on_watchlist_tickers(tmp_path):
    db_path = str(tmp_path / "e2e.db")
    tickers = ["AAPL", "MSFT", "NVDA"]

    result = pipeline_module.run_pipeline(tickers, db_path=db_path, lookback_days=10)

    assert len(result["events"]) == len(tickers)
    for event in result["events"]:
        assert event.event_type.value == "earnings"
        assert event.interpretation.confidence is not None
        assert event.interpretation.confirmation_score is not None
        # synthetic prices drift steadily upward -> should see a populated, positive 5m return
        assert event.market_reaction.return_5m is not None
        assert event.market_reaction.return_5m > 0

    assert isinstance(result["morning_brief"], list)
    assert len(result["morning_brief"]) == len(tickers)

    with connect(db_path) as conn:
        event_count = conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]
        evidence_count = conn.execute("SELECT COUNT(*) AS c FROM evidence").fetchone()["c"]
        alert_count = conn.execute("SELECT COUNT(*) AS c FROM alerts").fetchone()["c"]
        eval_count = conn.execute("SELECT COUNT(*) AS c FROM alert_evaluation").fetchone()["c"]

    assert event_count == len(tickers)
    assert evidence_count >= len(tickers)
    assert alert_count == eval_count
    assert alert_count > 0


def test_pipeline_is_idempotent_about_not_crashing_on_rerun(tmp_path):
    db_path = str(tmp_path / "e2e_rerun.db")
    tickers = ["AAPL"]

    first = pipeline_module.run_pipeline(tickers, db_path=db_path, lookback_days=10)
    second = pipeline_module.run_pipeline(tickers, db_path=db_path, lookback_days=10)

    assert len(first["events"]) == 1
    assert len(second["events"]) == 1
