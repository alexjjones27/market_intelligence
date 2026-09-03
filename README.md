# Market Intelligence (MVP)

Event-driven market intelligence and ranking system for US-listed equities.

This is **not** a news summarizer. Every event answers: what changed, which
assets are exposed, how different is it from expectations, and is the price
reaction large enough or persistent enough to matter. There is no trading
execution or autonomous buy/sell logic anywhere in this codebase.

`facts` (what was reported), `interpretation` (our derived scores), and
`market_reaction` (observed price/volume/IV response) are kept as separate,
independently-populated fields on every event so inference is never confused
with reported data. See `src/market_intel/models/event.py`.

## What's real vs. mocked

Every ingestor logs loudly (`WARNING`/`[MOCK]`) and every mocked row/table
carries an `is_mocked` flag when it's running on synthetic data. Nothing
fabricated is silently mixed in with real data without a flag.

| Source | Status | Notes |
|---|---|---|
| SEC EDGAR (8-K/10-Q/10-K) | **Real**, keyless | `ingestion/sec_edgar.py`. Needs `SEC_EDGAR_USER_AGENT` (required by SEC, not a secret). |
| Price / volume | **Real**, keyless | `ingestion/market_data.py`, calls Yahoo Finance's public chart endpoint directly with `requests` (not the `yfinance` package -- its bundled HTTP client didn't work through this environment's proxy; a plain `requests.get` against the same endpoint does). ~60-day intraday retention. |
| News | **Real** if `FINNHUB_API_KEY` set, else **mock** | `ingestion/news.py`. Default "one reliable feed" per the build spec. |
| Macro releases (CPI, PCE, NFP, unemployment, retail sales, GDP, jobless claims, Fed funds) | **Real** if `FRED_API_KEY` set, else **mock** | `ingestion/macro.py`. FRED has no consensus/expected-value field and no true point-in-time (ALFRED) vintages wired up -- see the module docstring for exactly what that means. |
| ISM Manufacturing PMI | **Always mock** | Licensed by ISM, not on FRED. No free source identified. |
| Earnings consensus (EPS/revenue actual vs. consensus) | **Always mock** | `ingestion/earnings.py`. **No provider was specified for this build**, so it ships a `MockEarningsProvider` behind an `EarningsConsensusProvider` interface. Ask before wiring one in (Finnhub `/stock/earnings`, Alpha Vantage `EARNINGS_ESTIMATES`, Zacks, IEX Cloud, ...). |
| Options implied volatility | **Always mock** | `ingestion/market_data.py:OptionsIVProvider`. No free/reliable IV source identified. Needs a paid vendor (CBOE DataShop, ORATS, Polygon.io options, Tradier). |
| Earnings calendar (forward-looking) | **Always mock** | `ingestion/calendar.py`. Random-ish date 10-90 days out per ticker. Real version should reuse the news provider's calendar endpoint. |
| Macro calendar | **Cadence heuristic, not a real feed** | `ingestion/calendar.py`. Assumes each release is roughly monthly (GDP quarterly). Real version should consume BLS/BEA/Fed's published calendars. |

Run `market-intel run` with no keys set and you'll see exactly which of the
above are mocked for that run, printed up front.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # fill in whatever keys you have; none are required to run
```

## CLI

```bash
market-intel run --tickers AAPL,MSFT,NVDA,JPM,XOM     # ingest -> score -> print ranked alerts
market-intel macro-regime                              # classify + persist the current growth/inflation regime
market-intel dashboard                                  # event study, sector heat map, calendar, CSV/parquet export
market-intel eval-summary                                # evaluation-framework rollup (precision, latency, returns)
market-intel init-db                                      # create/seed the DB only
```

`--tickers` defaults to `data/watchlist.json`. `--db-path` defaults to
`MARKET_INTEL_DB_PATH` in `.env` (or `./market_intel.db`).

## Architecture

```
ingestion/          Source layer. One module per source, common RawItem output
                     (source, publisher, pub/event timestamps, retrieval time,
                     confirmed flag). SEC EDGAR + price/volume are real;
                     everything else mocks behind a real interface -- see table above.

processing/
  entity_resolution.py   ticker/company resolution against the watchlist
  event_classification.py 8-K item codes + keyword rules -> one of 6 event types
  dedup.py                clusters raw items -> one Event with N Evidence records
  timestamps.py            UTC normalization + exchange-local time
  market_reaction.py       turns cached price snapshots into 5m/1h/1d/5d returns + volume z-score

scoring/            Five independent, unit-tested scoring functions:
  surprise.py             (actual-consensus)/|consensus|, z-scored vs. trailing history
  confidence.py            source confirmation + corroboration + extraction/timestamp certainty
  market_sensitivity.py    historical beta + sector membership (rate/commodity/FX exposure
                           stubs are explicit -- see module docstring)
  novelty.py                first-disclosure / guidance-shift / text-diff vs. prior N events
  market_confirmation.py    abnormal return + volume z + IV change + persistence
  macro_regime.py            2x2 growth x inflation regime grid

exposure/mapping.py  Static ticker -> sector -> {ETFs, peers, commodities, countries,
                     rate/FX sensitivity} table (data/sector_map.json). Extend by editing JSON.

alerts/
  alert_engine.py      immediate alerts (gated on surprise+confidence+confirmation
                       all clearing threshold) + morning brief (ranked, ungated)
  dashboard.py          event study, sector heat map, macro regime state, calendar, CSV/parquet export

evaluation/eval_log.py Logs every alert at generation time (latency filled in immediately),
                       updates outcomes (returns, volume/IV response, breadth, precision,
                       false-positive, PnL net of assumed costs) as they become observable.
                       `holdout` column enforces separating threshold-tuning data from eval data.

db/schema.sql        SQLite schema. facts/exposure/interpretation/market_reaction are JSON
                     columns on `events`; evidence is a proper one-to-many table. Point-in-time
                     tables (earnings_consensus, macro_releases) are insert-only -- a changed
                     value for an already-seen period is a new row, never an overwrite.

pipeline.py           Orchestrates all of the above; cli.py is a thin wrapper over it.
```

## Known MVP limitations (by design, not oversight)

- **No idempotent re-ingestion.** Re-running `market-intel run` against the
  same DB creates new `raw_items`/`events` rows rather than upserting against
  previously-seen filings/articles. Fine for a single demo run or a fresh DB
  per day; a production version needs content-based dedup on ingest.
- **`already_priced_in` confidence input is always 0.0.** There's no
  pre-event price-drift signal wired up yet to detect "this was already
  leaking into the price before the disclosure."
- **Guidance-direction extraction isn't implemented.** `facts.guidance_direction`
  stays null unless a future text-extraction step fills it in (LLM-assisted
  extraction from filing/press-release text, constrained to only write
  `facts`, never `interpretation`). Guidance-related scoring paths are all
  written to handle `None` gracefully.
- **Sector/market-sensitivity stubs** (`revenue_exposure_by_geography_stub`,
  `rate_duration_stub`, `index_passive_flow_relevance_stub` in
  `scoring/market_sensitivity.py`) return `None` rather than a fabricated
  number. Wire up real data sources before trusting these dimensions.
- **FRED macro data isn't true point-in-time (ALFRED).** See
  `ingestion/macro.py` module docstring for the precise approximation used.
- **Novelty/dedup near-duplicate detection uses `difflib`, not embeddings.**
  Deliberate per the build spec ("keep it simple for MVP"); fine at this
  scale, would need an upgrade for a broad, high-volume news feed.

## Tests

```bash
pytest -q   # 42 tests: dedup/clustering, surprise + confidence scoring,
            # market sensitivity/novelty/confirmation, and a fully-mocked
            # (no network) end-to-end pipeline run
```
