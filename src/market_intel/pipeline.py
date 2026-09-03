"""End-to-end pipeline: ingest -> normalize -> dedup -> enrich facts ->
score -> alert -> persist. This is the orchestration layer the CLI
drives; see cli.py for the thin command wrapper.
"""
from __future__ import annotations

import logging

from market_intel.alerts.alert_engine import (
    AlertThresholds,
    build_immediate_alerts,
    build_morning_brief,
    persist_alert,
)
from market_intel.db.database import connect, init_db, seed_static_tables
from market_intel.db.repository import (
    get_earnings_surprise_history,
    get_prior_events,
    insert_event,
    insert_raw_item,
)
from market_intel.evaluation.eval_log import log_alert_created
from market_intel.exposure.mapping import get_exposure_for_ticker, get_fx_sensitivity, get_sector_rate_sensitivity
from market_intel.ingestion.calendar import seed_earnings_calendar, seed_macro_calendar
from market_intel.ingestion.earnings import current_fiscal_period, get_provider, previous_fiscal_periods
from market_intel.ingestion.market_data import MarketDataIngestor, get_options_provider, persist_price_snapshots
from market_intel.ingestion.news import NewsIngestor
from market_intel.ingestion.sec_edgar import SecEdgarIngestor
from market_intel.models.event import Event, EventType
from market_intel.processing.data_quality import evaluate_data_quality
from market_intel.processing.dedup import cluster_and_build_events
from market_intel.processing.entity_resolution import load_watchlist, resolve_ticker
from market_intel.processing.event_classification import classify
from market_intel.processing.market_reaction import daily_return_for_date, populate_market_reaction
from market_intel.processing.timestamps import parse_utc
from market_intel.scoring.confidence import (
    EXPECTED_FACT_FIELDS,
    ConfidenceInputs,
    compute_confidence,
    extraction_completeness,
    needs_verification,
)
from market_intel.scoring.market_confirmation import abnormal_return, compute_confirmation_score, persistence_ratio
from market_intel.scoring.market_sensitivity import compute_beta, composite_market_sensitivity, returns_from_prices
from market_intel.scoring.novelty import compute_novelty_score
from market_intel.scoring.surprise import guidance_direction_label, guidance_direction_score, score_surprise

logger = logging.getLogger(__name__)

BENCHMARK_TICKER = "SPY"


def run_pipeline(
    tickers: list[str],
    db_path: str | None = None,
    lookback_days: int = 10,
    alert_thresholds: AlertThresholds | None = None,
    mode: str = "demo",
) -> dict:
    """`mode`: "live" refuses to substitute mock data for an unconfigured
    provider -- the corresponding fields are left null (an incomplete
    event) rather than silently filled with synthetic numbers. "backtest"
    and "demo" both allow the mock fallback, every such value flagged
    is_mocked=true throughout (see ingestion/*.py); they're distinguished
    only by intent, not behavior, in this MVP."""
    thresholds = alert_thresholds or AlertThresholds()
    init_db(db_path)
    seed_static_tables(db_path)

    with connect(db_path) as conn:
        raw_items = _ingest(tickers, lookback_days, mode)
        raw_items = _normalize(raw_items)
        for item in raw_items:
            insert_raw_item(conn, item)

        watchlist = load_watchlist()
        seed_earnings_calendar(conn, tickers, watchlist)
        seed_macro_calendar(conn)

        events = cluster_and_build_events(raw_items)
        logger.info("Clustered %d raw items into %d events", len(raw_items), len(events))

        market_data = MarketDataIngestor()
        options_provider = get_options_provider()
        earnings_provider = get_provider()
        price_cache: dict[str, list[dict]] = {}

        _prime_benchmark(conn, market_data, price_cache)

        for event in events:
            _enrich_facts(conn, event, earnings_provider, mode)
            event.exposure = get_exposure_for_ticker(event.ticker, conn)
            _populate_market_data(conn, event, market_data, options_provider, price_cache, mode)
            _score_event(conn, event)
            insert_event(conn, event)

        immediate_alerts = build_immediate_alerts(events, thresholds)
        morning_brief = build_morning_brief(events)

        for alert in immediate_alerts + morning_brief:
            persist_alert(conn, alert)
            log_alert_created(conn, alert["alert_id"], alert.get("_latency_seconds"))

    return {"events": events, "immediate_alerts": immediate_alerts, "morning_brief": morning_brief}


def _ingest(tickers: list[str], lookback_days: int, mode: str = "demo") -> list:
    sec = SecEdgarIngestor(lookback_days=lookback_days)
    news = NewsIngestor()
    items = []
    items.extend(sec.fetch(tickers))
    if mode == "live" and news.is_mocked:
        logger.warning("LIVE mode: news provider unconfigured (no FINNHUB_API_KEY) -- skipping, not mocking.")
        news_status = "skipped (live mode, unconfigured)"
    else:
        items.extend(news.fetch(tickers))
        news_status = "MOCKED" if news.is_mocked else "live"
    logger.info("Ingested %d raw items (SEC live, news %s)", len(items), news_status)
    return items


def _normalize(raw_items: list) -> list:
    watchlist = load_watchlist()
    resolved = []
    for item in raw_items:
        ticker = resolve_ticker(item.ticker_guess, item.company_guess, watchlist)
        if not ticker:
            logger.warning("Dropping unresolvable raw item: %r", item.title)
            continue
        item.ticker_guess = ticker
        if not item.company_guess:
            item.company_guess = watchlist[ticker]["company"]
        event_type = classify(item.event_type_guess, item.title, item.body_text)
        item.event_type_guess = event_type.value
        item.stabilize_id()
        resolved.append(item)
    return resolved


def _enrich_facts(conn, event: Event, earnings_provider, mode: str = "demo") -> None:
    if event.event_type != EventType.EARNINGS:
        return

    if mode == "live" and earnings_provider.is_mocked:
        logger.warning(
            "LIVE mode: earnings consensus provider unconfigured for %s -- leaving facts null, not mocking.",
            event.ticker,
        )
        return

    # Fiscal period must come from the REPORTED quarter/year end, not from
    # when the filing became public -- a quarter ending June 30 is
    # routinely filed in early August, which calendar-buckets to Q3 and
    # would look up the wrong quarter's consensus. Fall back to bucketing
    # the event timestamp only when no fiscal_period_end was captured
    # (e.g. a news-only event with no SEC filing evidence).
    fiscal_period_end = getattr(event.facts, "fiscal_period_end", None)
    if fiscal_period_end:
        period_dt = parse_utc(f"{fiscal_period_end}T00:00:00.000000Z")
    else:
        period_dt = parse_utc(event.timestamp_utc)
    period = current_fiscal_period(period_dt)

    # Seed a bit of trailing history so the surprise z-score has
    # something to standardize against (see db/repository.py).
    for prior_period in previous_fiscal_periods(period, 4):
        if not conn.execute(
            "SELECT 1 FROM earnings_consensus WHERE ticker = ? AND fiscal_period = ?", (event.ticker, prior_period)
        ).fetchone():
            consensus = earnings_provider.get_consensus(event.ticker, prior_period)
            if consensus:
                _persist_earnings_consensus(conn, event.ticker, prior_period, consensus)

    consensus = earnings_provider.get_consensus(event.ticker, period)
    if not consensus:
        return
    _persist_earnings_consensus(conn, event.ticker, period, consensus)

    event.facts.eps_actual = consensus.get("eps_actual")
    event.facts.eps_consensus = consensus.get("eps_consensus")
    event.facts.revenue_actual = consensus.get("revenue_actual")
    event.facts.revenue_consensus = consensus.get("revenue_consensus")
    if event.facts.guidance_direction is None:
        # MVP: no structured guidance extraction pipeline yet; left null
        # unless a downstream text-extraction step fills it in.
        pass


def _persist_earnings_consensus(conn, ticker: str, fiscal_period: str, consensus: dict) -> None:
    conn.execute(
        """
        INSERT INTO earnings_consensus (ticker, fiscal_period, eps_consensus, revenue_consensus,
            eps_actual, revenue_actual, source, is_mocked)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ticker, fiscal_period, consensus.get("eps_consensus"), consensus.get("revenue_consensus"),
            consensus.get("eps_actual"), consensus.get("revenue_actual"), consensus.get("source", "mock"),
            int(consensus.get("source") == "mock"),
        ),
    )


def _prime_benchmark(conn, market_data: MarketDataIngestor, cache: dict[str, list[dict]]) -> None:
    rows = market_data.fetch_daily(BENCHMARK_TICKER, range_="6mo")
    if rows:
        persist_price_snapshots(conn, rows)
        cache[BENCHMARK_TICKER] = rows
    intraday = market_data.fetch_intraday(BENCHMARK_TICKER, range_="5d", interval="5m")
    if intraday:
        persist_price_snapshots(conn, intraday)


def _ensure_daily_prices(conn, market_data: MarketDataIngestor, ticker: str, cache: dict[str, list[dict]]) -> list[dict]:
    if ticker in cache:
        return cache[ticker]
    rows = market_data.fetch_daily(ticker, range_="6mo")
    if rows:
        persist_price_snapshots(conn, rows)
    cache[ticker] = rows
    return rows


def _populate_market_data(conn, event: Event, market_data: MarketDataIngestor, options_provider, cache, mode: str = "demo") -> None:
    _ensure_daily_prices(conn, market_data, event.ticker, cache)
    intraday_rows = market_data.fetch_intraday(event.ticker, range_="5d", interval="5m")
    if intraday_rows:
        persist_price_snapshots(conn, intraday_rows)

    event.market_reaction = populate_market_reaction(conn, event.ticker, event.timestamp_utc)

    if mode == "live" and options_provider.is_mocked:
        logger.warning("LIVE mode: options IV provider unconfigured -- leaving iv_change null, not mocking.")
    else:
        event.market_reaction.iv_change = options_provider.get_iv_change(event.ticker, event.timestamp_utc)


def _score_event(conn, event: Event) -> None:
    interp = event.interpretation

    # --- Surprise ---------------------------------------------------
    # Per-component direction is computed from raw signs, never from the
    # blended composite -- an EPS miss alongside a revenue beat must
    # come out "mixed," not forced into a single positive/negative
    # label. See scoring/surprise.py:score_surprise.
    if event.event_type == EventType.EARNINGS:
        period = current_fiscal_period(parse_utc(event.timestamp_utc))
        eps_history = get_earnings_surprise_history(conn, event.ticker, "eps", exclude_period=period)
        rev_history = get_earnings_surprise_history(conn, event.ticker, "revenue", exclude_period=period)
        assessment = score_surprise(
            eps_actual=event.facts.eps_actual, eps_consensus=event.facts.eps_consensus,
            revenue_actual=event.facts.revenue_actual, revenue_consensus=event.facts.revenue_consensus,
            guidance_direction=event.facts.guidance_direction,
            eps_history=eps_history, revenue_history=rev_history,
        )
        interp.eps_surprise = assessment.eps_surprise
        interp.revenue_surprise = assessment.revenue_surprise
        interp.earnings_direction = assessment.direction
        interp.surprise_coverage = assessment.coverage
        interp.surprise_components_missing = assessment.components_missing
        interp.surprise_score = assessment.composite
    else:
        guidance_score = guidance_direction_score(event.facts.guidance_direction)
        interp.surprise_score = guidance_score
        interp.earnings_direction = guidance_direction_label(event.facts.guidance_direction)
        interp.surprise_coverage = 0.0 if guidance_score is None else 1.0
        interp.surprise_components_missing = [] if guidance_score is not None else ["guidance"]

    # --- Confidence ---------------------------------------------------
    primary_source_confirmed = any(e.confirmed for e in event.evidence)
    interp.primary_source_confirmed = primary_source_confirmed

    expected_fields = EXPECTED_FACT_FIELDS.get(event.event_type.value, [])
    facts_dict = event.facts.model_dump()
    completeness = extraction_completeness(facts_dict, expected_fields) if expected_fields else 0.5
    timestamp_certainty = 1.0 if event.source_tier.value == "primary" else 0.6
    opinion_fraction = 0.0 if event.source_tier.value == "primary" else 0.3

    confidence_inputs = ConfidenceInputs(
        primary_source_confirmed=primary_source_confirmed,
        num_corroborating_sources=len(event.evidence),
        extraction_completeness=completeness,
        timestamp_certainty=timestamp_certainty,
        # No pre-event drift signal is implemented yet (see
        # scoring/market_sensitivity.py stubs and README limitations).
        # None means "not assessed" -- compute_confidence drops this
        # component and renormalizes rather than defaulting to a value
        # that would misrepresent absence of evidence as evidence.
        already_priced_in=None,
        opinion_fraction=opinion_fraction,
    )
    interp.confidence = compute_confidence(confidence_inputs)
    interp.already_priced_in = confidence_inputs.already_priced_in

    # --- Market sensitivity --------------------------------------------
    rate_sensitivity = get_sector_rate_sensitivity(event.ticker, conn)
    asset_rows = conn.execute(
        "SELECT price FROM price_snapshots WHERE ticker = ? AND granularity = 'daily' ORDER BY timestamp_utc",
        (event.ticker,),
    ).fetchall()
    bench_rows = conn.execute(
        "SELECT price FROM price_snapshots WHERE ticker = ? AND granularity = 'daily' ORDER BY timestamp_utc",
        (BENCHMARK_TICKER,),
    ).fetchall()
    beta = None
    if asset_rows and bench_rows:
        n = min(len(asset_rows), len(bench_rows))
        asset_returns = returns_from_prices([r["price"] for r in asset_rows][-n:])
        bench_returns = returns_from_prices([r["price"] for r in bench_rows][-n:])
        beta = compute_beta(asset_returns, bench_returns)

    fx_sensitivity = get_fx_sensitivity(event.ticker, conn)
    sensitivity = composite_market_sensitivity(beta, rate_sensitivity, event.exposure.commodities, fx_sensitivity)
    interp.market_sensitivity_score = sensitivity.score
    interp.macro_sensitivity = sensitivity.label

    # --- Novelty ---------------------------------------------------
    prior_events = get_prior_events(conn, event.ticker, event.event_type.value, event.timestamp_utc, limit=5)
    prior_directions = [e.facts.guidance_direction for e in prior_events]
    current_text = event.evidence[0].source if event.evidence else None
    interp.novelty_score = compute_novelty_score(
        is_first_disclosure_of_type=len(prior_events) == 0,
        guidance_direction=event.facts.guidance_direction,
        most_recent_prior_guidance_direction=prior_directions[0] if prior_directions else None,
        current_text=current_text,
        prior_texts=[e.evidence[0].source for e in prior_events if e.evidence],
    )

    # --- Market confirmation ---------------------------------------
    event_date = parse_utc(event.timestamp_utc).date()
    benchmark_return_1d = daily_return_for_date(conn, BENCHMARK_TICKER, event_date)
    abn_return_1d = abnormal_return(event.market_reaction.return_1d, benchmark_return_1d)
    persistence = persistence_ratio(event.market_reaction.return_5m, event.market_reaction.return_1d)
    breadth = _sector_breadth(conn, event, event_date)

    confirmation = compute_confirmation_score(
        abnormal_return_1d=abn_return_1d,
        volume_z=event.market_reaction.volume_zscore,
        iv_change=event.market_reaction.iv_change,
        persistence_1d_over_5m=persistence,
        sector_breadth=breadth,
    )
    interp.market_confirmation_score = confirmation.score
    interp.market_confirmation_coverage = confirmation.coverage
    interp.market_confirmation_components_missing = confirmation.components_missing

    # --- Verification flag -------------------------------------------
    impact = max(abs(interp.surprise_score) if interp.surprise_score is not None else 0.0, confirmation.score)
    interp.verification_flag = needs_verification(interp.confidence, impact)

    # --- Data quality gate (see processing/data_quality.py) -----------
    event.data_quality = evaluate_data_quality(conn, event)


def _sector_breadth(conn, event: Event, event_date) -> float | None:
    peers = event.exposure.peers
    if not peers:
        return None
    ticker_return = daily_return_for_date(conn, event.ticker, event_date)
    if ticker_return is None:
        return None
    ticker_dir = 1 if ticker_return >= 0 else -1

    same, counted = 0, 0
    market_data = MarketDataIngestor()
    for peer in peers:
        rows = conn.execute(
            "SELECT 1 FROM price_snapshots WHERE ticker = ? AND granularity = 'daily' LIMIT 1", (peer,)
        ).fetchone()
        if not rows:
            fetched = market_data.fetch_daily(peer, range_="1mo")
            if fetched:
                persist_price_snapshots(conn, fetched)
        peer_return = daily_return_for_date(conn, peer, event_date)
        if peer_return is None:
            continue
        counted += 1
        if (1 if peer_return >= 0 else -1) == ticker_dir:
            same += 1
    return round(same / counted, 4) if counted else None
