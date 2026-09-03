-- Market Intelligence MVP schema (SQLite)
--
-- Design notes:
--   * `raw_items` holds every ingested item BEFORE clustering (one row per
--     press release / 8-K / news article / transcript). Deduplication turns
--     N raw_items into 1 event + N evidence rows.
--   * `events` keeps facts / exposure / interpretation / market_reaction as
--     separate JSON columns so an LLM's inference (interpretation) can never
--     be confused with reported data (facts) or observed data
--     (market_reaction). This mirrors the canonical event schema exactly.
--   * Point-in-time discipline: `earnings_consensus` and `macro_releases`
--     are insert-only for first-seen values; revisions get new rows, the
--     original is never overwritten. Backtests must always join on the
--     earliest `first_captured_at` / `first_published_at` row.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------
-- Watchlist & static mappings
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS watchlist (
    ticker      TEXT PRIMARY KEY,
    company     TEXT NOT NULL,
    cik         TEXT,
    sector      TEXT,
    active      INTEGER NOT NULL DEFAULT 1,
    added_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS sector_map (
    sector            TEXT PRIMARY KEY,
    sector_etfs       TEXT NOT NULL DEFAULT '[]',   -- JSON list
    commodities       TEXT NOT NULL DEFAULT '[]',   -- JSON list
    rate_sensitivity  TEXT,
    fx_sensitivity    TEXT NOT NULL DEFAULT '[]',   -- JSON list
    members           TEXT NOT NULL DEFAULT '{}'    -- JSON: ticker -> {peers, countries}
);

-- ---------------------------------------------------------------------
-- Source layer: raw ingested items (pre-clustering)
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS raw_items (
    raw_item_id         TEXT PRIMARY KEY,          -- uuid
    source               TEXT NOT NULL,             -- e.g. "SEC 8-K", "Finnhub News", "IR Release"
    source_tier          TEXT NOT NULL,              -- primary | professional | secondary
    publisher             TEXT,
    url                   TEXT,
    ticker_guess          TEXT,
    company_guess         TEXT,
    title                 TEXT,
    body_text             TEXT,
    event_type_guess      TEXT,
    published_at          TEXT,                       -- UTC ISO8601, as published
    event_timestamp_guess TEXT,                        -- UTC ISO8601, best guess of the actual event time
    exchange_local_time   TEXT,                        -- ISO8601 with offset, exchange-local
    retrieved_at          TEXT NOT NULL,               -- UTC ISO8601, when WE fetched it
    confirmed             INTEGER NOT NULL DEFAULT 0,  -- 1 = primary-source confirmed, 0 = unverified
    content_hash          TEXT,                        -- for near-dup detection
    is_mocked             INTEGER NOT NULL DEFAULT 0,  -- 1 if produced by a mock ingestor
    clustered_event_id    TEXT,                        -- set once assigned to an event
    created_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (clustered_event_id) REFERENCES events(event_id),
    -- Idempotent re-ingestion: re-running the pipeline against the same
    -- ticker/source/url must not create a duplicate row. (NULL url rows
    -- are exempt -- SQLite treats each NULL as distinct, which is fine:
    -- sources that never set a url don't get this protection, but none
    -- of the current ingestors omit it.)
    UNIQUE (ticker_guess, source, url)
);

CREATE INDEX IF NOT EXISTS idx_raw_items_ticker_time
    ON raw_items (ticker_guess, published_at);
CREATE INDEX IF NOT EXISTS idx_raw_items_unclustered
    ON raw_items (clustered_event_id);

-- ---------------------------------------------------------------------
-- Canonical events
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,         -- uuid
    event_type      TEXT NOT NULL,             -- earnings | guidance_change | m_and_a | exec_change | legal_regulatory | capital_allocation | other
    company         TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    timestamp_utc   TEXT NOT NULL,
    exchange_local_time TEXT,
    source_tier     TEXT NOT NULL,             -- highest tier among contributing evidence

    -- Separated sections. Facts = reported. Interpretation = derived scores.
    -- Market reaction = observed, populated asynchronously.
    facts           TEXT NOT NULL DEFAULT '{}',            -- JSON
    exposure        TEXT NOT NULL DEFAULT '{}',            -- JSON
    interpretation  TEXT NOT NULL DEFAULT '{}',            -- JSON
    market_reaction TEXT NOT NULL DEFAULT '{}',            -- JSON
    data_quality    TEXT NOT NULL DEFAULT '{"status": "pass", "issues": [], "blocking": false}',  -- JSON

    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_events_ticker_type_time
    ON events (ticker, event_type, timestamp_utc);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id   TEXT PRIMARY KEY,           -- uuid
    event_id      TEXT NOT NULL,
    source        TEXT NOT NULL,
    url           TEXT,
    publisher     TEXT,
    published_at  TEXT,
    retrieved_at  TEXT NOT NULL,
    confirmed     INTEGER NOT NULL DEFAULT 0,
    is_mocked     INTEGER NOT NULL DEFAULT 0,
    raw_item_id   TEXT,
    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
    FOREIGN KEY (raw_item_id) REFERENCES raw_items(raw_item_id)
);

CREATE INDEX IF NOT EXISTS idx_evidence_event ON evidence (event_id);

-- ---------------------------------------------------------------------
-- Earnings consensus (point-in-time; insert-only, never overwritten)
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS earnings_consensus (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker             TEXT NOT NULL,
    fiscal_period       TEXT NOT NULL,          -- e.g. "2026Q2"
    eps_consensus       REAL,
    revenue_consensus   REAL,
    eps_actual          REAL,
    revenue_actual       REAL,
    source               TEXT NOT NULL,
    is_mocked            INTEGER NOT NULL DEFAULT 0,
    first_captured_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_earnings_consensus_ticker
    ON earnings_consensus (ticker, fiscal_period, first_captured_at);

-- ---------------------------------------------------------------------
-- Macro calendar / releases (point-in-time; revisions are new rows)
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS macro_releases (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id          TEXT NOT NULL,          -- e.g. FRED series id "CPIAUCSL"
    release_name       TEXT NOT NULL,          -- e.g. "CPI"
    period             TEXT NOT NULL,          -- e.g. "2026-08"
    value               REAL NOT NULL,
    is_first_release    INTEGER NOT NULL DEFAULT 1,
    consensus           REAL,
    prior_period_value   REAL,
    published_at         TEXT NOT NULL,
    retrieved_at          TEXT NOT NULL,
    is_mocked             INTEGER NOT NULL DEFAULT 0,
    source                TEXT NOT NULL DEFAULT 'FRED'
);

CREATE INDEX IF NOT EXISTS idx_macro_releases_series_period
    ON macro_releases (series_id, period, published_at);

CREATE TABLE IF NOT EXISTS macro_regime_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of          TEXT NOT NULL,
    growth_state   TEXT NOT NULL,              -- up | down
    inflation_state TEXT NOT NULL,             -- up | down
    regime_label    TEXT NOT NULL,              -- reflation | overheat | stagflation | disinflation-slowdown ...
    expected_cross_asset_direction TEXT NOT NULL DEFAULT '{}'  -- JSON
);

-- ---------------------------------------------------------------------
-- Price / volume snapshots (cache for return & volume-zscore computation)
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS price_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    granularity   TEXT NOT NULL DEFAULT 'daily',  -- 'intraday' (5m bars) | 'daily'
    price         REAL NOT NULL,
    volume        REAL,
    is_mocked     INTEGER NOT NULL DEFAULT 0,
    UNIQUE (ticker, timestamp_utc, granularity)
);

CREATE INDEX IF NOT EXISTS idx_price_snapshots_ticker_time
    ON price_snapshots (ticker, timestamp_utc);

-- ---------------------------------------------------------------------
-- Alerts
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS alerts (
    alert_id           TEXT PRIMARY KEY,        -- uuid
    event_id           TEXT NOT NULL,
    alert_type         TEXT NOT NULL,            -- immediate | morning_brief
    generated_at       TEXT NOT NULL,
    published_at       TEXT,                     -- source publication time, for latency calc
    surprise_score     REAL,
    confidence_score    REAL,
    market_confirmation_score REAL,
    composite_rank_score REAL,
    headline             TEXT,
    expectation_gap       TEXT,
    reaction_summary       TEXT,
    assessment              TEXT,
    risk_note                TEXT,
    status_badge              TEXT,                      -- LIVE-primary | LIVE-secondary | MOCK | MIXED
    why_now                    TEXT NOT NULL DEFAULT '[]', -- JSON list
    sources                   TEXT NOT NULL DEFAULT '[]', -- JSON list of urls
    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_alerts_event ON alerts (event_id);
CREATE INDEX IF NOT EXISTS idx_alerts_generated_at ON alerts (generated_at);

-- ---------------------------------------------------------------------
-- Evaluation framework: outcome tracking per alert
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS alert_evaluation (
    alert_id            TEXT PRIMARY KEY,
    latency_seconds       REAL,                 -- publication -> alert generation
    return_5m              REAL,
    return_30m              REAL,
    return_1d                REAL,
    return_5d                 REAL,
    volume_zscore_response      REAL,
    iv_change_response           REAL,
    sector_breadth                 REAL,        -- fraction of sector peers/ETF moving same direction
    precision_label                 TEXT,        -- material | not_material | unknown, filled post-hoc
    false_positive                   INTEGER,     -- 1/0/NULL(unknown)
    pnl_net_of_costs_bps               REAL,
    holdout                              INTEGER NOT NULL DEFAULT 0, -- 1 if this alert is in the holdout period
    evaluated_at                          TEXT,
    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id) ON DELETE CASCADE
);

-- ---------------------------------------------------------------------
-- Macro calendar of upcoming releases (for the research dashboard)
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS macro_calendar (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    release_name  TEXT NOT NULL,
    series_id     TEXT,
    scheduled_at  TEXT NOT NULL,
    country       TEXT NOT NULL DEFAULT 'United States'
);

CREATE TABLE IF NOT EXISTS earnings_calendar (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT NOT NULL,
    company       TEXT,
    scheduled_at  TEXT NOT NULL,
    fiscal_period TEXT,
    confirmed     INTEGER NOT NULL DEFAULT 0,
    is_mocked     INTEGER NOT NULL DEFAULT 0
);
