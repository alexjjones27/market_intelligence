"""Daily OHLCV from Yahoo Finance's public chart endpoint.

We hit ``query1.finance.yahoo.com/v8/finance/chart`` directly with ``requests``
rather than going through the ``yfinance`` package: yfinance bundles curl_cffi
for TLS fingerprint impersonation, which does not survive this environment's
outbound proxy, while a plain GET against the same endpoint works and needs no
cookie/crumb handshake for chart data.

Adjustment policy (matters more than it looks, and is easy to get wrong):

Yahoo's chart endpoint already returns **split-adjusted** OHLC and volume in
``indicators.quote``; ``indicators.adjclose`` additionally strips dividends.
Verified directly: NVDA's unadjusted 2024-06-07 close was $1208.88 and the
endpoint reports $120.89 across a 10:1 split dated 2024-06-10.

So:

* ``open/high/low/close/volume`` are used **exactly as returned** -- already
  split-consistent with each other, and still on the price scale the market
  actually traded at (dividends not stripped). That is what Appendix A.3 needs,
  since its alphas compare CLOSE against MAX(HIGH, 20) and divide it by EPS.
  Re-scaling close by ``adj_close/close`` would silently push a *dividend*
  adjustment into highs and lows and desynchronise them from the fundamentals.
* ``adj_close`` (split + dividend adjusted) is used for **returns only**, so a
  4:1 split is not booked as a -75% day and dividend-paying names are not
  systematically penalised versus non-payers.

All series are returned already aligned to a common trading calendar with **no
forward-filling of prices across missing days** -- a filled price would create
a fake zero return and flatter every volatility alpha.
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

logger = logging.getLogger(__name__)

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_FIELDS = ("open", "high", "low", "close", "adj_close", "volume")
SPLIT_URL = CHART_URL  # same endpoint, queried with a coarse interval


@dataclass(frozen=True)
class PriceData:
    """Wide price panels: index=trading date, columns=ticker."""

    frames: dict[str, pd.DataFrame]
    failed: list[str]

    def __getitem__(self, field: str) -> pd.DataFrame:
        return self.frames[field]

    @property
    def tickers(self) -> list[str]:
        return list(self.frames["close"].columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.frames["close"].index

    @property
    def returns(self) -> pd.DataFrame:
        """Simple daily total return from the adjusted close."""
        return self.frames["adj_close"].pct_change(fill_method=None)


def _fetch_one(session: requests.Session, ticker: str, start: str, end: str) -> pd.DataFrame:
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(pd.Timestamp(end).timestamp()) + 86400
    resp = session.get(
        CHART_URL.format(ticker=ticker),
        params={"period1": p1, "period2": p2, "interval": "1d", "events": "div,split"},
        headers={"User-Agent": "Mozilla/5.0 (alpha_lab research)"},
        timeout=45,
    )
    resp.raise_for_status()
    payload = resp.json()
    result = (payload.get("chart") or {}).get("result")
    if not result:
        raise ValueError(f"empty chart payload for {ticker}")
    res = result[0]
    ts = res.get("timestamp") or []
    if not ts:
        raise ValueError(f"no bars for {ticker}")

    quote = res["indicators"]["quote"][0]
    adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose")

    df = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None),
    )
    df["adj_close"] = adj if adj is not None else df["close"]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])

    # No further adjustment: Yahoo's quote block is already split-consistent.
    # Guard against the pathological case where adj_close is wildly off close
    # (seen on a few thin tickers), which would corrupt the return series.
    ratio = (df["adj_close"] / df["close"]).replace([np.inf, -np.inf], np.nan)
    bad = ratio.notna() & ((ratio > 1.05) | (ratio < 0.5))
    if bad.any():
        logger.warning(
            "%s: %d bars with implausible adj_close/close ratio; using close for those",
            ticker, int(bad.sum()),
        )
        df.loc[bad, "adj_close"] = df.loc[bad, "close"]

    return df[["open", "high", "low", "close", "adj_close", "volume"]]


def load_prices(
    cfg: Config,
    tickers: list[str],
    refresh: bool = False,
    extra_tickers: tuple[str, ...] = (),
) -> PriceData:
    """Fetch daily bars for ``tickers``, cached one parquet per ticker."""
    cache = Cache(cfg.cache_path / "prices")
    session = requests.Session()
    start, end = cfg.data.start, cfg.data.end
    wanted = list(dict.fromkeys(list(tickers) + list(extra_tickers)))

    per_ticker: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    for i, tkr in enumerate(wanted):
        key = f"{tkr}_{start}_{end}"

        def produce(t=tkr) -> pd.DataFrame:
            time.sleep(cfg.data.request_sleep_s)
            return _fetch_one(session, t, start, end)

        try:
            df = cache.frame(key, produce, refresh=refresh)
        except Exception as exc:
            logger.warning("price fetch failed for %s: %s", tkr, exc)
            failed.append(tkr)
            continue
        if df is None or df.empty:
            failed.append(tkr)
            continue
        per_ticker[tkr] = df
        if (i + 1) % 50 == 0:
            logger.info("prices: %d/%d fetched (%d failed)", i + 1, len(wanted), len(failed))

    if not per_ticker:
        raise RuntimeError("no price data fetched for any ticker")

    # A shared calendar: union of all observed session dates. Names that were
    # not listed/traded on a date stay NaN rather than being filled.
    calendar = pd.DatetimeIndex(sorted(set().union(*(df.index for df in per_ticker.values()))))
    calendar = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]

    frames = {
        field: pd.DataFrame(
            {t: df[field].reindex(calendar) for t, df in per_ticker.items()},
            index=calendar,
        ).sort_index(axis=1)
        for field in _FIELDS
    }
    logger.info(
        "prices: %d tickers x %d sessions (%d failed)", len(per_ticker), len(calendar), len(failed)
    )
    return PriceData(frames=frames, failed=failed)


def apply_liquidity_filter(prices: PriceData, cfg: Config) -> pd.DataFrame:
    """Boolean tradeability mask: price and dollar-volume floors.

    Applied as a *mask*, not a row drop, so the shape of every alpha panel stays
    identical and nothing silently realigns.
    """
    close = prices["close"]
    dollar_volume = close * prices["volume"]
    mask = (close >= cfg.data.min_price) & (dollar_volume >= cfg.data.min_dollar_volume)
    return mask.fillna(False)


def _fetch_splits(session: requests.Session, ticker: str) -> pd.DataFrame:
    """Split events for one ticker: ex-date and ratio.

    Queried at monthly granularity over the full history, which keeps the
    payload tiny -- we only want the ``events.splits`` block, not the bars.
    """
    resp = session.get(
        CHART_URL.format(ticker=ticker),
        params={"range": "max", "interval": "1mo", "events": "split"},
        headers={"User-Agent": "Mozilla/5.0 (alpha_lab research)"},
        timeout=45,
    )
    resp.raise_for_status()
    result = (resp.json().get("chart") or {}).get("result")
    if not result:
        return pd.DataFrame(columns=["ex_date", "ratio"])
    splits = (result[0].get("events") or {}).get("splits") or {}
    rows = []
    for ev in splits.values():
        denom = float(ev.get("denominator") or 0)
        numer = float(ev.get("numerator") or 0)
        if denom <= 0 or numer <= 0:
            continue
        rows.append(
            {
                "ex_date": pd.to_datetime(ev["date"], unit="s", utc=True)
                .tz_convert("America/New_York")
                .normalize()
                .tz_localize(None),
                "ratio": numer / denom,
            }
        )
    return pd.DataFrame(rows, columns=["ex_date", "ratio"]).sort_values("ex_date")


def load_split_factors(
    cfg: Config, tickers: list[str], trading_days: pd.DatetimeIndex, refresh: bool = False
) -> pd.DataFrame:
    r"""Per-ticker factor that puts an as-reported per-share figure on the same
    share basis as this panel's split-adjusted prices.

    Why this is needed at all: Yahoo reports prices **retroactively
    split-adjusted** (Apple's 2019 close reads ~\$50, not the ~\$198 it traded
    at), while SEC XBRL reports per-share fundamentals **as filed** (Apple's
    FY2019 diluted EPS is \$11.89, in pre-split shares). Divide one by the other
    and Apple shows a 24% earnings yield in 2019 -- a P/E of 4 for a company
    that actually traded near 17.

    With ``cum(t)`` the product of every split ratio with an ex-date at or
    before ``t``, this returns ``factor(t) = cum(t) / cum(T_end)``, matching
    Yahoo's own convention ``P_adj(t) = P_raw(t) * cum(t) / cum(T_end)``.
    A per-share figure filed on date ``f`` is then multiplied by ``factor(f)``;
    a **share count** filed on ``f`` is divided by it, since share counts move
    the opposite way through a split. Market cap, being their product, comes out
    invariant -- which is the arithmetic check that the convention is right.
    """
    cache = Cache(cfg.cache_path / "splits")
    session = requests.Session()

    factors: dict[str, pd.Series] = {}
    n_with_splits = 0
    for ticker in tickers:
        def produce(t=ticker) -> pd.DataFrame:
            time.sleep(cfg.data.request_sleep_s)
            return _fetch_splits(session, t)

        try:
            events = cache.frame(f"{ticker}_splits", produce, refresh=refresh)
        except Exception as exc:
            logger.warning("split fetch failed for %s: %s", ticker, exc)
            events = pd.DataFrame(columns=["ex_date", "ratio"])

        factor = pd.Series(1.0, index=trading_days)
        if events is not None and not events.empty:
            events = events.copy()
            events["ex_date"] = pd.to_datetime(events["ex_date"])
            relevant = events[events["ex_date"] <= trading_days.max()]
            if not relevant.empty:
                cum = pd.Series(1.0, index=trading_days)
                for _, ev in relevant.iterrows():
                    cum.loc[cum.index >= ev["ex_date"]] *= ev["ratio"]
                factor = cum / cum.iloc[-1]
                if (relevant["ex_date"] >= trading_days.min()).any():
                    n_with_splits += 1
        factors[ticker] = factor

    logger.info(
        "split factors: %d/%d tickers split inside the panel window",
        n_with_splits, len(tickers),
    )
    return pd.DataFrame(factors, index=trading_days).sort_index(axis=1)
