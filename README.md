# Market Intelligence (MVP)

Event-driven market intelligence and ranking system for US-listed equities.

This is **not** a news summarizer. Every event answers: what changed, which
assets are exposed, how different is it from expectations, and is the price
reaction large enough or persistent enough to matter. There is no trading
execution or autonomous buy/sell logic anywhere in this codebase.

`facts` (what was reported), `interpretation` (our derived scores),
`market_reaction` (observed price/volume/IV response), and `data_quality`
(pre-scoring integrity checks) are kept as separate, independently-populated
fields on every event so inference is never confused with reported data. See
`src/market_intel/models/event.py`.

## Companion subsystem: `alpha_lab` (quant backtest replication)

This repository also carries **`alpha_lab`**, a separate subsystem that
replicates and stress-tests the three-stage framework from Kou et al.,
*Automate Strategy Finding with LLM in Quant Investment* (Findings of EMNLP
2025), on point-in-time S&P 500 data. It shares nothing with the event pipeline
above except the repository -- different package (`src/alpha_lab/`), different
dependencies (`requirements-alpha-lab.txt`), different tests
(`tests/alphalab/`).

See **[ALPHA_LAB.md](ALPHA_LAB.md)** for the design, the data provenance, the
leakage audit, and what the replication found about the paper's own
construction. Results live in [reports/RESULTS.md](reports/RESULTS.md).

Like the rest of this repository, it contains no trading execution and no
autonomous buy/sell logic: it is a backtest and a measurement harness.

## Data integrity, not just data source honesty (v2)

A round of review against real pipeline output found that "the data sources
are honestly labeled" wasn't the same as "the output can't mislead." Two
concrete bugs and several design gaps got fixed as a result -- worth stating
plainly rather than burying in a changelog:

- **A real bug:** 10-Q/10-K event timestamps were set from SEC's
  `reportDate` (the fiscal quarter/year END) instead of the filing date. An
  earnings event's canonical timestamp could land a month before the market
  actually learned the numbers, corrupting the market-reaction window it was
  scored against. Fixed in `ingestion/sec_edgar.py`; fiscal period end is now
  captured separately (`facts.fiscal_period_end`) and used only for
  consensus lookup, never as the event's disclosure time.
- **A real bug:** a single blended surprise score forced every earnings
  event into "positive" or "negative," even when EPS missed and revenue beat
  in the same release. `scoring/surprise.py:score_surprise` now classifies
  direction from each component's sign independently -- `positive | negative
  | mixed | inconclusive | unknown` -- and the blended composite still
  exists, but only for ranking, never for the label.
- **A design gap:** nothing validated a fact pattern before it was allowed
  to compete for an alert. `processing/data_quality.py` now runs a gate
  before scoring: scale-mismatched reported values, duplicate evidence,
  evidence spans too wide to trust as one disclosure, consensus figures that
  weren't actually known before the event, and insufficient earnings data
  all get flagged (`warning`, surfaced everywhere; `reject`, quarantined
  from immediate alerts) rather than silently scored as if clean.
- **A design gap:** "confirmation" conflated source quality with market
  reaction. It was already market-only under the hood (abnormal return,
  volume, IV, persistence) but named ambiguously; it's now
  `market_confirmation_score`, with `primary_source_confirmed` promoted to
  its own explicit gate condition, independent of confidence.
- **A design gap:** `already_priced_in` defaulted to `0.0` ("not priced
  in"), which is indistinguishable from "we checked and found no evidence of
  it." It's now `bool | None`; `None` ("not assessed") drops out of the
  confidence weighting and renormalizes over what's actually known, instead
  of quietly claiming more precision than the evidence supports.
- **A correctness fix in clustering:** two unrelated SEC filings a month
  apart could match on near-duplicate title text alone if they shared
  generic wording (same company, same 8-K item codes) -- clustering them as
  "corroborating" evidence for the same event when they weren't. Filing
  dates are now baked into SEC-sourced titles, and a hard 14-day outer bound
  caps how far the near-duplicate override can reach regardless of text
  similarity (`processing/dedup.py:MAX_NEAR_DUPLICATE_GAP`).
- **Idempotency:** re-running the pipeline against the same DB no longer
  accumulates duplicate `raw_items`/`events` rows. `raw_item_id` and
  `event_id` are now deterministic (uuid5, keyed off ticker/source/url and
  ticker/type/date respectively), `raw_items` re-inserts are ignored on
  conflict, and `events` upserts. Not perfectly stable in every case --
  see Limitations -- but a rerun on the same window is a no-op, not a
  duplicate.
- **A raised bar for what counts as alertable.** Immediate-alert thresholds
  moved from `min_confidence=0.5` / `min_confirmation=0.25` to
  `min_confidence=0.75` / `min_market_confirmation=0.65`, and
  `primary_source_confirmed=True` + `data_quality.status != "reject"` are
  now hard requirements by default (`--allow-secondary-source` to relax the
  first). **A consequence you'll see immediately: the demo watchlist
  produces zero immediate alerts most runs.** That's not a bug -- it's
  what happens when a run is honestly scored against a fully-mocked
  consensus provider (every earnings event picks up a
  `consensus_not_point_in_time` warning) and the price reaction alone isn't
  enough to clear the bar. A strong price move confirms the market reacted;
  it doesn't confirm a mocked number was true.
- **`--mode live`**: unconfigured providers (news, earnings consensus,
  options IV) now produce an **incomplete event** -- the field stays null,
  logged as a warning -- instead of silently substituting mock data. Mock
  fallback is still the default (`--mode demo`) for exercising the full
  pipeline without any keys.

## What's real vs. mocked

Every ingestor logs loudly (`WARNING`/`[MOCK]`) and every mocked row/table
carries an `is_mocked` flag when it's running on synthetic data. Nothing
fabricated is silently mixed in with real data without a flag -- and in
`--mode live`, nothing fabricated gets mixed in at all (see above).

| Source | Status | Notes |
|---|---|---|
| SEC EDGAR (8-K/10-Q/10-K) | **Real**, keyless | `ingestion/sec_edgar.py`. Needs `SEC_EDGAR_USER_AGENT` (required by SEC, not a secret). |
| Price / volume | **Real**, keyless | `ingestion/market_data.py`, calls Yahoo Finance's public chart endpoint directly with `requests` (not the `yfinance` package -- its bundled HTTP client didn't work through this environment's proxy; a plain `requests.get` against the same endpoint does). ~60-day intraday retention. |
| News | **Real** if `FINNHUB_API_KEY` set, else **mock** | `ingestion/news.py`. Default "one reliable feed" per the build spec. |
| Macro releases (CPI, PCE, NFP, unemployment, retail sales, GDP, jobless claims, Fed funds) | **Real** if `FRED_API_KEY` set, else **mock** | `ingestion/macro.py`. FRED has no consensus/expected-value field and no true point-in-time (ALFRED) vintages wired up -- see the module docstring for exactly what that means. |
| ISM Manufacturing PMI | **Always mock** | Licensed by ISM, not on FRED. No free source identified. |
| Earnings consensus (EPS/revenue actual vs. consensus) | **Always mock** | `ingestion/earnings.py`. **No provider was specified for this build**, so it ships a `MockEarningsProvider` behind an `EarningsConsensusProvider` interface. Ask before wiring one in (Finnhub `/stock/earnings`, Alpha Vantage `EARNINGS_ESTIMATES`, Zacks, IEX Cloud, ...). Every event scored against it carries a `consensus_not_point_in_time` data-quality warning -- see below. |
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
market-intel run --tickers AAPL,MSFT,NVDA,JPM,XOM         # ingest -> score -> print ranked alerts
market-intel run --mode live                                # unconfigured providers -> null fields, never mocked
market-intel macro-regime                                    # classify + persist the current growth/inflation regime
market-intel dashboard                                        # event study, sector heat map, calendar, CSV/parquet export
market-intel quarantine                                        # events flagged by the data-quality gate (warning/reject)
market-intel eval-summary                                        # evaluation-framework rollup (precision, latency, returns)
market-intel init-db                                              # create/seed the DB only
```

`--tickers` defaults to `data/watchlist.json`. `--db-path` defaults to
`MARKET_INTEL_DB_PATH` in `.env` (or `./market_intel.db`). `run` also takes
`--min-surprise`, `--min-confidence`, `--min-market-confirmation`, and
`--allow-secondary-source` to tune the immediate-alert gate.

## Architecture

```
ingestion/          Source layer. One module per source, common RawItem output
                     (source, publisher, pub/event timestamps, retrieval time,
                     confirmed flag). SEC EDGAR + price/volume are real;
                     everything else mocks behind a real interface -- see table above.

processing/
  entity_resolution.py   ticker/company resolution against the watchlist
  event_classification.py 8-K item codes + keyword rules -> one of 6 event types
  dedup.py                clusters raw items -> one Event with N Evidence records;
                          hard outer bound on near-duplicate matching so generic
                          titles can't merge unrelated filings across quarters
  data_quality.py          pre-scoring integrity gate: scale mismatches, duplicate
                           evidence, inconsistent event spans, non-point-in-time
                           consensus, insufficient data -> warning/reject
  timestamps.py             UTC normalization + exchange-local time
  market_reaction.py        turns cached price snapshots into 5m/1h/1d/5d returns + volume z-score

scoring/            Five independent, unit-tested scoring functions:
  surprise.py             per-component EPS/revenue surprise, z-scored vs. trailing
                          history; direction (positive/negative/mixed/inconclusive/
                          unknown) computed from component signs, never the blend
  confidence.py            source confirmation + corroboration + extraction/timestamp
                           certainty + three-state already-priced-in (renormalizes
                           when unassessed rather than defaulting to a value)
  market_sensitivity.py    historical beta + sector membership (rate/commodity/FX exposure
                           stubs are explicit -- see module docstring)
  novelty.py                first-disclosure / guidance-shift / text-diff vs. prior N events
  market_confirmation.py    abnormal return + volume z + IV change + persistence,
                           market-observed only; coverage + missing-components tracked
  macro_regime.py            2x2 growth x inflation regime grid; historical-tendency
                            framing, never presented as a forecast

exposure/mapping.py  Static ticker -> sector -> {ETFs, peers, commodities, countries,
                     rate/FX sensitivity} table (data/sector_map.json). Extend by editing JSON.

alerts/
  alert_engine.py      immediate alerts -- gated on primary_source_confirmed,
                       data_quality != reject, confidence >= 0.75, market_confirmation
                       >= 0.65, and surprise clearing threshold where applicable --
                       + morning brief (ranked, ungated). Every alert carries a
                       status_badge (LIVE-primary/LIVE-secondary/MOCK/MIXED) and a
                       why_now list of the specific triggers that fired.
  dashboard.py          event study, sector heat map, macro regime state, calendar,
                       quarantine queue, CSV/parquet export

evaluation/eval_log.py Logs every alert at generation time (latency filled in immediately),
                       updates outcomes (returns, volume/IV response, breadth, precision,
                       false-positive, PnL net of assumed costs) as they become observable.
                       `holdout` column enforces separating threshold-tuning data from eval data.

db/schema.sql        SQLite schema. facts/exposure/interpretation/market_reaction/data_quality
                     are JSON columns on `events`; evidence is a proper one-to-many table
                     with a UNIQUE constraint on raw_items for idempotent re-ingestion.
                     Point-in-time tables (earnings_consensus, macro_releases) are
                     insert-only -- a changed value for an already-seen period is a
                     new row, never an overwrite.

pipeline.py           Orchestrates all of the above; cli.py is a thin wrapper over it.
```

## Known MVP limitations (by design, not oversight)

- **Event-id stability isn't perfect.** `event_id` is deterministic
  (ticker + event type + date bucket), so a rerun over an unchanged window
  is a true no-op. But if a later run discovers an evidence item with an
  earlier timestamp than any seen before, the cluster's earliest-item
  anchor can shift the date bucket and mint a new id. True stability needs
  a persisted cluster identity (e.g. keyed off the primary filing's
  accession number) -- not implemented.
- **`already_priced_in` has no real signal source yet.** The three-state
  field exists and is handled correctly (renormalizes when `None`), but
  nothing currently computes pre-event drift (1d/5d/20d abnormal return,
  IV change, news volume, estimate revisions before the event) to set it to
  `True`/`False` in the first place -- it stays `None` throughout this MVP.
- **Guidance-direction extraction isn't implemented.** `facts.guidance_direction`
  stays null unless a future text-extraction step fills it in (LLM-assisted
  extraction from filing/press-release text, constrained to only write
  `facts`, never `interpretation`). Guidance-related scoring paths are all
  written to handle `None` gracefully.
- **Sector/market-sensitivity stubs** (`revenue_exposure_by_geography_stub`,
  `rate_duration_stub`, `index_passive_flow_relevance_stub` in
  `scoring/market_sensitivity.py`) return `None` rather than a fabricated
  number. Wire up real data sources before trusting these dimensions.
- **The exposure map is a flat static table, not a graph.** It distinguishes
  direct issuer from sector/peers/commodities/FX, but doesn't distinguish
  *observed* exposure (what's in `data/sector_map.json`) from *inferred*
  relationships (customer/supplier chains, index membership) -- because it
  doesn't attempt the latter at all yet. A real read-through graph with that
  observed/inferred/hypothetical distinction is future work, not a small fix.
- **FRED macro data isn't true point-in-time (ALFRED).** See
  `ingestion/macro.py` module docstring for the precise approximation used.
- **No purged/embargoed evaluation splits or baselines.** `evaluation/eval_log.py`
  has a `holdout` column and logs everything needed, but nothing yet
  enforces time-based purging/embargo windows around events, and there's no
  benchmark comparison (random alert, raw surprise alone, momentum alone) to
  check the composite system actually beats simple alternatives out of
  sample. Needed before trusting any precision number this framework
  produces at scale.
- **No analyst feedback loop.** There's no mechanism yet for a researcher to
  mark an alert material/immaterial or correct/incorrect, separate from the
  market-outcome evaluation.
- **Novelty/dedup near-duplicate detection uses `difflib`, not embeddings.**
  Deliberate per the build spec ("keep it simple for MVP"); fine at this
  scale, would need an upgrade for a broad, high-volume news feed.

## Tests

```bash
pytest -q   # 63 tests: dedup/clustering (incl. the near-duplicate outer-bound
            # fix), every scoring module (incl. mixed/inconclusive direction
            # classification and all 4 macro-regime quadrants), and a
            # fully-mocked (no network) end-to-end pipeline run that also
            # checks rerun idempotency
```
