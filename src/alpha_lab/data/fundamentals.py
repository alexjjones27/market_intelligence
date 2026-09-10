"""Point-in-time fundamentals from SEC EDGAR's XBRL ``companyfacts`` API.

This is the module that decides whether the whole backtest is honest.

Every XBRL fact SEC returns carries both the *period* it describes (``start``/
``end``) and the date it was actually **filed**. A backtest that keys
fundamentals on the fiscal period end is reading Apple's September-quarter
earnings roughly 30 days before the market saw them -- a look-ahead worth
several percent a year on any value or growth factor. Everything here is keyed
on ``filed`` and nothing else. A value enters the panel on its filing date and
not one session earlier.

Design notes:

* **Flows** (revenue, EPS, cash flow) are accumulated to a trailing-twelve-month
  figure so quarterly seasonality does not masquerade as a signal, and so the
  fiscal-Q4 gap (10-Ks often tag only the annual figure) does not leave holes.
  A TTM value becomes known on the filing date of its *most recent* constituent
  quarter. Four quarters are only summed if they actually span 330-400 days, so
  a missing quarter produces NaN instead of a silently wrong three-quarter sum.
* **Instants** (assets, equity, debt, share count) become known on their filing
  date; when one filing reports several balance-sheet dates, the most recent
  period end wins.
* **Restatements**: the first-filed value for a period is used. At any date the
  panel therefore holds what the market actually had, not what the figure was
  later revised to.
* **Staleness**: a fundamental is carried forward from its filing date but
  expires after ``max_staleness_days``. Without that, a company that stops
  filing keeps a three-year-old EPS forever and its valuation alphas drift into
  fiction.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from alpha_lab.config import Config
from alpha_lab.data.cache import Cache
from alpha_lab.data.http import get_with_retry

logger = logging.getLogger(__name__)

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# US-GAAP tag preferences, most-specific first. Companies tag the same economic
# quantity under different concepts depending on industry and filing vintage,
# so each field takes the first tag that yields usable facts.
FLOW_CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "cogs": ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold", "CostOfServices"],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "dividends_per_share": [
        "CommonStockDividendsPerShareDeclared",
        "CommonStockDividendsPerShareCashPaid",
    ],
    "interest_expense": ["InterestExpense", "InterestExpenseDebt", "InterestAndDebtExpense"],
    "depreciation_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
}

INSTANT_CONCEPTS: dict[str, list[str]] = {
    "total_assets": ["Assets"],
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "short_term_debt": ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "inventory": ["InventoryNet"],
    "receivables": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "payables": ["AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "shares_outstanding": [
        "dei:EntityCommonStockSharesOutstanding",
        "CommonStockSharesOutstanding",
        "CommonStockSharesIssued",
    ],
}

# Per-share figures are filed on the share basis of their filing date, while
# prices are retroactively split-adjusted. These must be put on the same basis
# or every valuation ratio is wrong by the post-filing split factor. Per-share
# amounts scale one way through a split, share counts the other.
PER_SHARE_FIELDS = frozenset({"eps", "dividends_per_share"})
SHARE_COUNT_FIELDS = frozenset({"shares_outstanding"})

QUARTER_DAYS = (80, 100)
ANNUAL_DAYS = (350, 380)
TTM_SPAN_DAYS = (330, 400)
MAX_STALENESS_DAYS = 400

# Derived fields the alpha library expects but XBRL does not tag directly.
DERIVED_FIELDS = (
    "total_debt", "book_value", "ebit", "ebitda", "market_cap", "enterprise_value",
    "book_value_per_share", "revenue_per_share", "cash_flow_per_share",
)


@dataclass(frozen=True)
class FundamentalData:
    """Wide fundamental panels: index=trading date, columns=ticker.

    Every value in every frame was publicly filed on or before its row date.
    """

    frames: dict[str, pd.DataFrame]
    missing: list[str]
    coverage: pd.DataFrame

    def __getitem__(self, field: str) -> pd.DataFrame:
        return self.frames[field]

    def get(self, field: str, like: pd.DataFrame) -> pd.DataFrame:
        """Fetch a field, or an all-NaN frame shaped like ``like`` if absent."""
        if field in self.frames:
            return self.frames[field]
        return pd.DataFrame(np.nan, index=like.index, columns=like.columns)


def _pick_unit(units: dict) -> list[dict]:
    """Choose the unit series with the most observations, preferring money."""
    if not units:
        return []
    for preferred in ("USD", "USD/shares", "shares", "pure"):
        if preferred in units and units[preferred]:
            return units[preferred]
    return max(units.values(), key=len)


def _facts_frame(payload: dict, tag: str) -> pd.DataFrame:
    """Extract one concept as a tidy frame, or empty if the tag is absent."""
    taxonomy, _, name = tag.partition(":")
    if not name:
        taxonomy, name = "us-gaap", tag
    block = payload.get("facts", {}).get(taxonomy, {}).get(name)
    if not block:
        return pd.DataFrame()
    facts = _pick_unit(block.get("units", {}))
    if not facts:
        return pd.DataFrame()
    df = pd.DataFrame(facts)
    if "filed" not in df or "end" not in df or "val" not in df:
        return pd.DataFrame()
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["start"] = pd.to_datetime(df["start"], errors="coerce") if "start" in df else pd.NaT
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    return df.dropna(subset=["end", "filed", "val"])


def _derive_missing_quarters(
    quarters: pd.DataFrame, annuals: pd.DataFrame
) -> pd.DataFrame:
    """Reconstruct the un-tagged fiscal Q4 from the annual figure.

    Most large filers tag Q1-Q3 quarterly in their 10-Qs but report only the
    full year in the 10-K, so the fourth quarter never appears as a ~90-day
    fact. Without it the four-quarter TTM window can never close on contiguous
    quarters, and every valuation and growth alpha ends up refreshing once a
    year instead of four times -- which both dulls the signal and misdates it.

    Where an annual fact covers exactly three tagged quarters, the fourth is
    ``annual - sum(three)``, and it becomes known on the 10-K's filing date
    (which is precisely when the market could have derived it too).
    """
    if quarters.empty or annuals.empty:
        return quarters
    derived = []
    for _, yr in annuals.iterrows():
        inside = quarters[
            (quarters["start"] >= yr["start"] - pd.Timedelta(days=5))
            & (quarters["end"] <= yr["end"] + pd.Timedelta(days=5))
        ]
        inside = inside.drop_duplicates("end")
        if len(inside) != 3:
            continue
        covered = int(inside["days"].sum())
        if not 250 <= covered <= 290:
            continue
        # The gap is whichever ~90-day slice of the fiscal year is untagged.
        ends = sorted(inside["end"])
        starts = sorted(inside["start"])
        if starts[0] > yr["start"] + pd.Timedelta(days=5):
            gap_start, gap_end = yr["start"], starts[0] - pd.Timedelta(days=1)
        else:
            gap_start, gap_end = ends[-1] + pd.Timedelta(days=1), yr["end"]
        gap_days = (gap_end - gap_start).days
        if not QUARTER_DAYS[0] - 15 <= gap_days <= QUARTER_DAYS[1] + 15:
            continue
        derived.append(
            {
                "start": gap_start,
                "end": gap_end,
                "val": yr["val"] - inside["val"].sum(),
                # Known only when the 10-K lands, never at the fiscal period end.
                "filed": yr["filed"],
                "days": gap_days,
            }
        )
    if not derived:
        return quarters
    return pd.concat([quarters, pd.DataFrame(derived)], ignore_index=True)


def _flow_known_series(df: pd.DataFrame) -> pd.Series:
    """Build a TTM step-function: index = date first known, value = TTM figure."""
    if df.empty or "start" not in df or df["start"].isna().all():
        return pd.Series(dtype=float)
    d = df.dropna(subset=["start"]).copy()
    d["days"] = (d["end"] - d["start"]).dt.days

    annual = d[d["days"].between(*ANNUAL_DAYS)]
    annual_first = (
        annual.sort_values("filed").drop_duplicates("end", keep="first")
        if not annual.empty
        else annual
    )

    ttm_series = pd.Series(dtype=float)
    quarters = d[d["days"].between(*QUARTER_DAYS)]
    quarters = _derive_missing_quarters(
        quarters.sort_values("filed").drop_duplicates("end", keep="first"), annual_first
    )
    if len(quarters) >= 4:
        # First disclosure per fiscal period: what the market actually had.
        q = (
            quarters.sort_values("filed")
            .drop_duplicates("end", keep="first")
            .sort_values("end")
            .reset_index(drop=True)
        )
        value = q["val"].rolling(4).sum()
        # Guard against a missing quarter silently producing a 3-quarter sum.
        span = (q["end"] - q["start"].shift(3)).dt.days
        # A TTM figure is knowable only once all four of its quarters are filed.
        # (Rolling a max over datetimes is unsupported in pandas, and casting to
        # int64 silently mixes microsecond and nanosecond epochs, so take the
        # window maximum directly.)
        filed = q["filed"].to_numpy()
        known = pd.Series(
            [filed[max(0, i - 3) : i + 1].max() for i in range(len(q))], index=q.index
        )
        ok = value.notna() & span.between(*TTM_SPAN_DAYS)
        ok.iloc[:3] = False  # first three rows have no complete window
        if ok.any():
            ttm_series = pd.Series(
                value[ok].to_numpy(), index=pd.DatetimeIndex(known[ok].to_numpy())
            )

    annual_series = pd.Series(dtype=float)
    if not annual_first.empty:
        annual_series = pd.Series(
            annual_first["val"].to_numpy(),
            index=pd.DatetimeIndex(annual_first["filed"].to_numpy()),
        )

    def _dedupe(s: pd.Series) -> pd.Series:
        if s.empty:
            return s
        s = s.sort_index()
        return s[~s.index.duplicated(keep="last")]

    ttm_series, annual_series = _dedupe(ttm_series), _dedupe(annual_series)
    if ttm_series.empty:
        return annual_series
    if annual_series.empty:
        return ttm_series
    # TTM is the preferred series; the annual figure only fills dates where no
    # TTM value had yet been published (typically the first year of history).
    merged = pd.concat([annual_series, ttm_series])
    merged = merged.sort_index()
    combined = ttm_series.reindex(merged.index.unique()).sort_index()
    combined = combined.combine_first(annual_series.reindex(combined.index))
    return _dedupe(combined.dropna())


def _instant_known_series(df: pd.DataFrame) -> pd.Series:
    """Balance-sheet step-function keyed on filing date."""
    if df.empty:
        return pd.Series(dtype=float)
    d = df.copy()
    if "start" in d:
        d = d[d["start"].isna()] if d["start"].notna().any() else d
    if d.empty:
        return pd.Series(dtype=float)
    # One filing can report several balance-sheet dates; the latest wins.
    d = d.sort_values(["filed", "end"]).drop_duplicates("filed", keep="last")
    s = pd.Series(d["val"].values, index=pd.to_datetime(d["filed"].values)).sort_index()
    return s[~s.index.duplicated(keep="last")]


def _to_trading_days(
    series: pd.Series, trading_days: pd.DatetimeIndex, max_staleness_days: int
) -> pd.Series:
    """Forward-fill a known-on step-function onto the trading calendar.

    Forward-fill only, and bounded: a figure survives ``max_staleness_days``
    past its filing date and then expires to NaN.
    """
    if series.empty:
        return pd.Series(np.nan, index=trading_days)
    s = series.sort_index()
    idx = s.index.union(trading_days)
    filled = s.reindex(idx).ffill().reindex(trading_days)
    age = pd.Series(
        trading_days.values,
        index=trading_days,
    ) - pd.Series(s.index.values, index=s.index).reindex(idx).ffill().reindex(trading_days)
    stale = age.dt.days > max_staleness_days
    return filled.mask(stale)



def _apply_split_basis(
    series: pd.Series, field: str, ticker: str, split_factors: pd.DataFrame | None
) -> pd.Series:
    """Restate an as-filed per-share figure onto the panel's split-adjusted basis.

    Applied at the *filing* date, before the value is forward-filled: each
    filing reports on its own share basis, so the correction belongs to the
    filing, not to the day it happens to be read.
    """
    if split_factors is None or ticker not in split_factors.columns:
        return series
    if field not in PER_SHARE_FIELDS and field not in SHARE_COUNT_FIELDS:
        return series

    factor_series = split_factors[ticker]
    idx = factor_series.index.union(series.index)
    at_filing = factor_series.reindex(idx).ffill().bfill().reindex(series.index)
    at_filing = at_filing.where(at_filing > 0).fillna(1.0)

    if field in PER_SHARE_FIELDS:
        return series * at_filing
    return series / at_filing


def _extract_ticker_facts(payload: dict) -> pd.DataFrame:
    """Reduce a ~4MB companyfacts blob to a compact known-date/value table."""
    rows: list[pd.DataFrame] = []
    for field, tags in FLOW_CONCEPTS.items():
        for tag in tags:
            series = _flow_known_series(_facts_frame(payload, tag))
            if not series.empty:
                rows.append(pd.DataFrame({"field": field, "known": series.index, "val": series.values}))
                break
    for field, tags in INSTANT_CONCEPTS.items():
        for tag in tags:
            series = _instant_known_series(_facts_frame(payload, tag))
            if not series.empty:
                rows.append(pd.DataFrame({"field": field, "known": series.index, "val": series.values}))
                break
    if not rows:
        return pd.DataFrame(columns=["field", "known", "val"])
    return pd.concat(rows, ignore_index=True)


def load_fundamentals(
    cfg: Config,
    ciks: pd.Series,
    trading_days: pd.DatetimeIndex,
    refresh: bool = False,
    split_factors: pd.DataFrame | None = None,
) -> FundamentalData:
    """Fetch SEC facts for each ticker and align onto the trading calendar.

    ``ciks`` maps ticker -> zero-padded CIK. Only the compact extracted table is
    cached (a few KB per name); the multi-MB raw JSON is parsed and discarded.
    """
    cache = Cache(cfg.cache_path / "fundamentals")
    session = requests.Session()
    headers = {"User-Agent": cfg.data.sec_user_agent, "Accept-Encoding": "gzip, deflate"}

    per_ticker: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for i, (ticker, cik) in enumerate(ciks.dropna().items()):
        def produce(c=cik) -> pd.DataFrame:
            time.sleep(cfg.data.request_sleep_s)
            resp = get_with_retry(
                session, COMPANYFACTS_URL.format(cik=c), headers=headers, timeout=60
            )
            return _extract_ticker_facts(resp.json())

        try:
            df = cache.frame(f"{ticker}_{cik}", produce, refresh=refresh)
        except Exception as exc:
            logger.warning("fundamentals fetch failed for %s (CIK %s): %s", ticker, cik, exc)
            missing.append(ticker)
            continue
        if df is None or df.empty:
            missing.append(ticker)
            continue
        per_ticker[ticker] = df
        if (i + 1) % 50 == 0:
            logger.info("fundamentals: %d/%d (%d missing)", i + 1, len(ciks), len(missing))

    fields = sorted(set(FLOW_CONCEPTS) | set(INSTANT_CONCEPTS))
    frames: dict[str, pd.DataFrame] = {}
    for field in fields:
        cols = {}
        for ticker, df in per_ticker.items():
            sub = df[df["field"] == field]
            if sub.empty:
                continue
            s = pd.Series(sub["val"].values, index=pd.to_datetime(sub["known"].values))
            s = s.sort_index()
            s = s[~s.index.duplicated(keep="last")]
            s = _apply_split_basis(s, field, ticker, split_factors)
            cols[ticker] = _to_trading_days(s, trading_days, MAX_STALENESS_DAYS)
        frames[field] = (
            pd.DataFrame(cols, index=trading_days).sort_index(axis=1)
            if cols
            else pd.DataFrame(index=trading_days)
        )

    coverage = pd.DataFrame(
        {
            "field": fields,
            "n_tickers": [frames[f].shape[1] for f in fields],
            "pct_cells_present": [
                100.0 * frames[f].notna().to_numpy().mean() if frames[f].size else 0.0
                for f in fields
            ],
        }
    )
    logger.info(
        "fundamentals: %d tickers parsed, %d missing", len(per_ticker), len(missing)
    )
    return FundamentalData(frames=frames, missing=missing, coverage=coverage)


def add_derived_fields(
    fundamentals: FundamentalData, close: pd.DataFrame
) -> FundamentalData:
    """Compute the composite fields A.3 needs but XBRL does not tag."""
    f = dict(fundamentals.frames)
    like = close

    def g(name: str) -> pd.DataFrame:
        frame = f.get(name)
        if frame is None or frame.empty:
            return pd.DataFrame(np.nan, index=like.index, columns=like.columns)
        return frame.reindex(index=like.index, columns=like.columns)

    lt, st = g("long_term_debt"), g("short_term_debt")
    # A missing current portion is far more often 'not tagged' than 'zero', but
    # treating it as zero here would understate leverage for exactly the names
    # that tag sloppily. Sum only where at least one leg is present.
    f["total_debt"] = lt.add(st, fill_value=0.0).where(lt.notna() | st.notna())

    equity = g("total_equity")
    f["book_value"] = equity

    op_income = g("operating_income")
    f["ebit"] = op_income
    f["ebitda"] = op_income.add(g("depreciation_amortization"), fill_value=0.0).where(op_income.notna())

    shares = g("shares_outstanding")
    f["market_cap"] = close * shares
    f["enterprise_value"] = f["market_cap"].add(f["total_debt"], fill_value=0.0).sub(
        g("cash"), fill_value=0.0
    ).where(f["market_cap"].notna())

    safe_shares = shares.where(shares > 0)
    f["book_value_per_share"] = equity / safe_shares
    f["revenue_per_share"] = g("revenue") / safe_shares
    f["cash_flow_per_share"] = g("operating_cash_flow") / safe_shares

    # gross_profit is absent for many financials; reconstruct where possible.
    gp, rev, cogs = g("gross_profit"), g("revenue"), g("cogs")
    f["gross_profit"] = gp.where(gp.notna(), rev - cogs)

    coverage = pd.DataFrame(
        {
            "field": sorted(f),
            "n_tickers": [f[k].shape[1] for k in sorted(f)],
            "pct_cells_present": [
                100.0 * f[k].notna().to_numpy().mean() if f[k].size else 0.0 for k in sorted(f)
            ],
        }
    )
    return FundamentalData(frames=f, missing=fundamentals.missing, coverage=coverage)
