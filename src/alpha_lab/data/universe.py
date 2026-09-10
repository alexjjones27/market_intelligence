"""Point-in-time S&P 500 membership.

Survivorship bias is the single easiest way to manufacture a fake backtest edge
in a US equity study, so this module goes out of its way to avoid it.

Rather than taking today's constituent list and pretending it held in 2019
(which silently deletes every company that was dropped for underperforming, or
went bankrupt, or got acquired), we fetch the *Wikipedia revision that was live
on each snapshot date* and parse the constituent table as it stood then. A
stock is tradeable on date ``t`` only if it appears in the most recent snapshot
at or before ``t``.

Caveats, stated plainly:

* Wikipedia edits lag real index changes by days, occasionally weeks. Membership
  is therefore approximately, not exactly, point-in-time. The error is a few
  days of inclusion/exclusion at the margin of a ~500-name universe.
* Snapshots are taken at a configurable frequency (quarterly by default), so an
  add-then-drop inside one snapshot interval is missed.
* Delisted tickers get *no price history from Yahoo*, which reintroduces
  survivorship bias through the back door. :func:`membership_coverage_report`
  quantifies exactly how many name-days that costs, so the effect is measured
  rather than assumed away.
"""
from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import requests

from alpha_lab.config import Config
from alpha_lab.data.cache import Cache
from alpha_lab.data.http import get_with_retry

logger = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_INDEX = "https://en.wikipedia.org/w/index.php"
WIKI_TITLE = "List of S&P 500 companies"

# Wikipedia's API answers rapid bursts with 429. One request per second is
# comfortably inside its unauthenticated budget and the results are cached.
WIKI_SLEEP_S = 1.2

# Wikipedia writes class-B share tickers with a dot; Yahoo uses a hyphen.
_TICKER_FIXES = {"BRK.B": "BRK-B", "BF.B": "BF-B", "BF.A": "BF-A", "GEV.WI": "GEV"}


def normalize_ticker(raw: str) -> str:
    t = str(raw).strip().upper()
    t = _TICKER_FIXES.get(t, t)
    return t.replace(".", "-")


@dataclass(frozen=True)
class UniverseData:
    """Point-in-time membership plus the static metadata we need downstream."""

    membership: pd.DataFrame
    """Boolean, index=snapshot_date, columns=ticker. True == in the index."""

    sectors: pd.Series
    """ticker -> GICS sector. Used for the sector-neutrality decomposition."""

    ciks: pd.Series
    """ticker -> zero-padded 10-digit CIK, for SEC lookups."""

    snapshot_dates: list[pd.Timestamp]

    @property
    def tickers(self) -> list[str]:
        return sorted(self.membership.columns)

    def members_on(self, date: pd.Timestamp) -> list[str]:
        """Names in the index as of ``date`` (most recent snapshot at or before)."""
        valid = self.membership.index[self.membership.index <= pd.Timestamp(date)]
        if len(valid) == 0:
            return []
        row = self.membership.loc[valid.max()]
        return sorted(row.index[row].tolist())

    def daily_mask(self, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
        """Expand snapshots to a daily boolean mask via as-of (backward) fill.

        Backward fill only: a snapshot dated 2021-03-31 governs 2021-03-31
        onward, never earlier. That direction matters -- forward-filling would
        leak future index membership into the past.
        """
        m = self.membership.reindex(
            self.membership.index.union(trading_days)
        ).sort_index()
        m = m.ffill().reindex(trading_days)
        return m.fillna(False).astype(bool)


def _snapshot_dates(start: str, end: str, freq: str = "QE") -> list[pd.Timestamp]:
    dates = pd.date_range(start=start, end=end, freq=freq).tolist()
    first = pd.Timestamp(start)
    if not dates or dates[0] > first:
        dates = [first] + dates
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    return [d for d in dates if d <= today]


def _fetch_revision_id(session: requests.Session, ts: pd.Timestamp, ua: str) -> int | None:
    """Newest revision at or before ``ts``."""
    resp = get_with_retry(
        session,
        WIKI_API,
        params={
            "action": "query",
            "prop": "revisions",
            "titles": WIKI_TITLE,
            "rvlimit": 1,
            "rvstart": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rvdir": "older",
            "rvprop": "ids|timestamp",
            "format": "json",
        },
        headers={"User-Agent": ua},
        timeout=45,
    )
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        revs = page.get("revisions") or []
        if revs:
            return int(revs[0]["revid"])
    return None


def _find_column(columns: list[str], *candidates: str) -> str | None:
    """First column whose lowercased name contains any candidate substring."""
    lowered = [(c, str(c).strip().lower()) for c in columns]
    for cand in candidates:
        for original, low in lowered:
            if cand in low:
                return original
    return None


def _parse_constituent_table(html: str) -> pd.DataFrame:
    """Pull the constituents table out of a Wikipedia revision.

    The page's column layout has drifted over the years -- the ticker column was
    'Ticker symbol' until 2019 and 'Symbol' after, a 'SEC filings' column came
    and went, and 'GICS Sub Industry' gained a hyphen -- so the table is
    identified by the columns we actually need rather than by position or by an
    exact header match.
    """
    tables = pd.read_html(io.StringIO(html))
    for tbl in tables:
        cols = [str(c).strip() for c in tbl.columns]
        has_symbol = _find_column(cols, "ticker symbol", "symbol", "ticker") is not None
        has_sector = _find_column(cols, "gics sector") is not None
        if has_symbol and has_sector:
            out = tbl.copy()
            out.columns = cols
            return out
    raise ValueError("no constituents table found in revision HTML")


def _fetch_snapshot(session: requests.Session, ts: pd.Timestamp, ua: str) -> pd.DataFrame:
    revid = _fetch_revision_id(session, ts, ua)
    if revid is None:
        raise ValueError(f"no Wikipedia revision at or before {ts.date()}")
    resp = get_with_retry(
        session, WIKI_INDEX, params={"oldid": revid}, headers={"User-Agent": ua}, timeout=60
    )
    tbl = _parse_constituent_table(resp.text)

    cols = list(tbl.columns)
    symbol_col = _find_column(cols, "ticker symbol", "symbol", "ticker")
    sector_col = _find_column(cols, "gics sector")
    cik_col = _find_column(cols, "cik")

    out = pd.DataFrame(
        {
            "snapshot_date": ts,
            "ticker": tbl[symbol_col].map(normalize_ticker),
            "sector": tbl[sector_col].astype(str).str.strip(),
        }
    )
    if cik_col is not None:
        out["cik"] = (
            pd.to_numeric(tbl[cik_col], errors="coerce")
            .astype("Int64")
            .map(lambda v: f"{int(v):010d}" if pd.notna(v) else None)
        )
    else:
        out["cik"] = None
    return out.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"])


def load_universe(cfg: Config, refresh: bool = False, freq: str = "QE") -> UniverseData:
    """Build (and cache) point-in-time membership over the configured span."""
    cache = Cache(cfg.cache_path / "universe")
    session = requests.Session()
    ua = cfg.data.sec_user_agent
    dates = _snapshot_dates(cfg.data.start, cfg.data.end, freq=freq)

    frames: list[pd.DataFrame] = []
    failures: list[pd.Timestamp] = []
    for ts in dates:
        def produce(t=ts) -> pd.DataFrame:
            time.sleep(WIKI_SLEEP_S)
            return _fetch_snapshot(session, t, ua)

        try:
            frames.append(cache.frame(f"snapshot_{ts.date()}", produce, refresh=refresh))
        except Exception as exc:
            logger.warning("universe snapshot %s failed: %s", ts.date(), exc)
            failures.append(ts)

    if not frames:
        raise RuntimeError("could not fetch any universe snapshot")
    if failures:
        # A missed snapshot is not fatal -- the previous one is carried forward
        # by the as-of fill -- but it widens the window in which membership is
        # stale, so it is surfaced rather than swallowed.
        logger.warning(
            "%d/%d universe snapshots unavailable (%s); membership between them "
            "is carried forward from the last good snapshot",
            len(failures), len(dates), ", ".join(str(f.date()) for f in failures[:5]),
        )
    snaps = pd.concat(frames, ignore_index=True)
    snaps["snapshot_date"] = pd.to_datetime(snaps["snapshot_date"])

    if not cfg.universe.point_in_time:
        # Survivorship-biased mode, kept only so the bias can be *measured*.
        latest = snaps.loc[snaps["snapshot_date"] == snaps["snapshot_date"].max()]
        snaps = pd.concat(
            [latest.assign(snapshot_date=d) for d in snaps["snapshot_date"].unique()],
            ignore_index=True,
        )
        logger.warning(
            "universe.point_in_time=False: today's constituents applied to all "
            "history. This IS survivorship bias and results are not trustworthy."
        )

    membership = (
        snaps.assign(member=True)
        .pivot_table(index="snapshot_date", columns="ticker", values="member", aggfunc="first")
        .fillna(False)
        .astype(bool)
        .sort_index()
    )

    if cfg.data.max_universe_size:
        keep = membership.sum(axis=0).sort_values(ascending=False)
        keep = keep.head(cfg.data.max_universe_size).index
        membership = membership[sorted(keep)]
        logger.warning(
            "max_universe_size=%s truncates the universe to the %s most "
            "persistent members -- itself a survivorship filter. Smoke runs only.",
            cfg.data.max_universe_size,
            cfg.data.max_universe_size,
        )

    latest_meta = snaps.sort_values("snapshot_date").drop_duplicates("ticker", keep="last")
    latest_meta = latest_meta.set_index("ticker")
    sectors = latest_meta["sector"].reindex(membership.columns)
    ciks = latest_meta["cik"].reindex(membership.columns)

    return UniverseData(
        membership=membership,
        sectors=sectors,
        ciks=ciks,
        snapshot_dates=list(membership.index),
    )


def membership_coverage_report(universe: UniverseData, priced: set[str]) -> pd.DataFrame:
    """How much of the point-in-time universe we actually have prices for.

    Yahoo drops history for delisted tickers, so names that left the index via
    bankruptcy or acquisition frequently return nothing. Every such name is a
    hole that biases results *upward* (the losers vanish). This report is what
    lets the write-up state the size of that hole instead of hand-waving it.
    """
    rows = []
    for date, row in universe.membership.iterrows():
        members = set(row.index[row])
        have = members & priced
        rows.append(
            {
                "snapshot_date": date,
                "n_members": len(members),
                "n_with_prices": len(have),
                "n_missing": len(members - have),
                "pct_missing": 100.0 * len(members - have) / max(len(members), 1),
                "missing_tickers": ",".join(sorted(members - have)[:25]),
            }
        )
    return pd.DataFrame(rows)
