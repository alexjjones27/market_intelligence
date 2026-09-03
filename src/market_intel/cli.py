"""CLI entry point.

    market-intel run                    # ingest + score + print ranked alerts
    market-intel dashboard              # research dashboard + CSV/parquet export
    market-intel eval-summary           # evaluation framework rollup
    market-intel init-db                # create/seed the database only
"""
from __future__ import annotations

import logging

import click

from market_intel.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@click.group()
def cli():
    """Event-driven market intelligence system (MVP)."""


def _default_tickers() -> list[str]:
    from market_intel.processing.entity_resolution import load_watchlist

    return list(load_watchlist().keys())


@cli.command()
@click.option("--db-path", default=None, help="SQLite DB path (defaults to config).")
def init_db(db_path):
    from market_intel.db.database import init_db as _init_db
    from market_intel.db.database import seed_static_tables

    _init_db(db_path)
    seed_static_tables(db_path)
    click.echo(f"Initialized and seeded database at {db_path or settings.db_path}")


@cli.command()
@click.option("--tickers", default=None, help="Comma-separated tickers (defaults to watchlist.json).")
@click.option("--lookback-days", default=10, show_default=True, help="SEC EDGAR filing lookback window.")
@click.option("--db-path", default=None, help="SQLite DB path (defaults to config).")
@click.option("--min-surprise", default=0.25, show_default=True)
@click.option("--min-confidence", default=0.5, show_default=True)
@click.option("--min-confirmation", default=0.25, show_default=True)
def run(tickers, lookback_days, db_path, min_surprise, min_confidence, min_confirmation):
    """Run the full pipeline end-to-end and print ranked alerts."""
    from market_intel.alerts.alert_engine import AlertThresholds
    from market_intel.pipeline import run_pipeline

    ticker_list = [t.strip().upper() for t in tickers.split(",")] if tickers else _default_tickers()
    click.echo(f"Running pipeline for: {', '.join(ticker_list)}")
    _warn_mocked_sources()

    thresholds = AlertThresholds(min_surprise=min_surprise, min_confidence=min_confidence, min_confirmation=min_confirmation)
    result = run_pipeline(ticker_list, db_path=db_path, lookback_days=lookback_days, alert_thresholds=thresholds)

    click.echo(f"\n{len(result['events'])} events built from raw ingestion.\n")
    _print_alerts("IMMEDIATE ALERTS", result["immediate_alerts"])
    _print_alerts("MORNING BRIEF (ranked, all events)", result["morning_brief"])


def _print_alerts(title: str, alerts: list[dict]) -> None:
    click.echo("=" * 78)
    click.echo(title)
    click.echo("=" * 78)
    if not alerts:
        click.echo("(none)")
        return
    for i, a in enumerate(alerts, 1):
        click.echo(f"\n[{i}] {a['headline']}")
        click.echo(
            f"    rank={a['composite_rank_score']:.3f}  surprise={_fmt(a['surprise_score'])}  "
            f"confidence={_fmt(a['confidence_score'])}  confirmation={_fmt(a['confirmation_score'])}"
        )
        click.echo(f"    Expectation gap: {a['expectation_gap']}")
        click.echo(f"    Reaction:        {a['reaction_summary']}")
        click.echo(f"    Assessment:      {a['assessment']}")
        click.echo(f"    Risk:            {a['risk_note']}")
        if a.get("sources"):
            click.echo(f"    Sources:         {', '.join(a['sources'][:3])}")
    click.echo()


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"


@cli.command()
@click.option("--db-path", default=None, help="SQLite DB path (defaults to config).")
@click.option("--export-dir", default=None, help="Directory to write events.csv / events.parquet.")
def dashboard(db_path, export_dir):
    """Print the research dashboard (calendar, event study, sector heat
    map, macro regime) and export the flat events table."""
    from pathlib import Path

    from market_intel.config import EXPORTS_DIR
    from market_intel.db.database import connect
    from market_intel.alerts.dashboard import (
        event_study_by_type,
        export_events_csv,
        export_events_parquet,
        macro_regime_state,
        sector_heat_map,
        upcoming_event_calendar,
    )

    export_dir = Path(export_dir) if export_dir else EXPORTS_DIR
    export_dir.mkdir(parents=True, exist_ok=True)

    with connect(db_path) as conn:
        click.echo("EVENT STUDY BY TYPE")
        for row in event_study_by_type(conn):
            click.echo(f"  {row['event_type']:<20} n={row['count']:<4} avg_1d={_fmt(row['avg_return_1d'])} avg_5d={_fmt(row['avg_return_5d'])}")

        click.echo("\nSECTOR HEAT MAP (avg 1d return)")
        for row in sector_heat_map(conn):
            click.echo(f"  {row['sector']:<20} {row['avg_return_1d']:+.2%}  (n={row['n_events']})")

        click.echo("\nMACRO REGIME STATE")
        regime = macro_regime_state(conn)
        click.echo(f"  {regime}" if regime else "  (no regime data yet -- run `market-intel macro-regime`)")

        click.echo("\nUPCOMING CALENDAR")
        for row in upcoming_event_calendar(conn, limit=20):
            click.echo(f"  {row['scheduled_at']}  [{row['type']}] {row['label']}")

        csv_path = export_events_csv(conn, export_dir / "events.csv")
        parquet_path = export_events_parquet(conn, export_dir / "events.parquet")
        click.echo(f"\nExported: {csv_path}\nExported: {parquet_path}")


@cli.command()
@click.option("--db-path", default=None, help="SQLite DB path (defaults to config).")
def macro_regime(db_path):
    """Fetch macro releases, classify the current growth/inflation
    regime (2x2 grid), and persist it for the dashboard."""
    from datetime import datetime, timezone

    from market_intel.db.database import connect, dumps
    from market_intel.ingestion.macro import MacroIngestor, persist_releases
    from market_intel.scoring.macro_regime import classify_regime_from_releases

    ingestor = MacroIngestor()
    if ingestor.is_mocked:
        click.echo("NOTE: FRED_API_KEY not set -- macro releases are MOCKED.")
    releases = ingestor.fetch_releases()

    with connect(db_path) as conn:
        persist_releases(conn, releases)
        by_name = {r["release_name"]: r for r in releases}
        assessment = classify_regime_from_releases(by_name)
        if assessment is None:
            click.echo("Not enough data to classify a regime.")
            return
        conn.execute(
            "INSERT INTO macro_regime_history (as_of, growth_state, inflation_state, regime_label, expected_cross_asset_direction) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                assessment.growth_state, assessment.inflation_state, assessment.regime_label,
                dumps(assessment.expected_cross_asset_direction),
            ),
        )

    click.echo(f"Regime: {assessment.regime_label} (growth={assessment.growth_state}, inflation={assessment.inflation_state})")
    click.echo(f"Expected cross-asset direction: {assessment.expected_cross_asset_direction}")


@cli.command()
@click.option("--db-path", default=None, help="SQLite DB path (defaults to config).")
def eval_summary(db_path):
    """Print the evaluation-framework rollup (precision, latency, returns by horizon)."""
    from market_intel.db.database import connect
    from market_intel.evaluation.eval_log import evaluation_summary

    with connect(db_path) as conn:
        summary = evaluation_summary(conn)
    for k, v in summary.items():
        click.echo(f"{k}: {v}")


def _warn_mocked_sources() -> None:
    mocked = []
    if settings.news_is_mocked:
        mocked.append("news (set FINNHUB_API_KEY)")
    if settings.earnings_is_mocked:
        mocked.append("earnings consensus (EARNINGS_PROVIDER=mock)")
    if settings.macro_is_mocked:
        mocked.append("macro releases (set FRED_API_KEY)")
    if settings.options_is_mocked:
        mocked.append("options IV (OPTIONS_PROVIDER=mock)")
    if mocked:
        click.echo("NOTE: running with MOCKED sources -> " + "; ".join(mocked))


if __name__ == "__main__":
    cli()
