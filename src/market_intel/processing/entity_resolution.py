"""Ticker / company entity resolution against the watchlist.

MVP scope: our ingestors are already per-ticker (SEC EDGAR and news are
both fetched with a known ticker as the query parameter), so resolution
is mostly a validation + company-name-fallback step rather than a full
NER pipeline. This keeps the module honest about what it does; a future
broad-scan news feed (not filtered by ticker) would need real NER/fuzzy
matching against a much larger universe, which is out of scope here.
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path

from market_intel.config import settings


def load_watchlist(watchlist_path: str | None = None) -> dict[str, dict]:
    """Return {ticker: {company, cik, sector}}."""
    path = watchlist_path or settings.watchlist_path
    data = json.loads(Path(path).read_text())
    return {e["ticker"]: e for e in data.get("tickers", [])}


def resolve_ticker(
    ticker_guess: str | None,
    company_guess: str | None,
    watchlist: dict[str, dict],
    fuzzy_threshold: float = 0.82,
) -> str | None:
    """Resolve a raw item to a watchlist ticker, or None if it can't be
    confidently resolved (caller should drop / quarantine such items)."""
    if ticker_guess and ticker_guess.upper() in watchlist:
        return ticker_guess.upper()

    if company_guess:
        names = {t: e["company"] for t, e in watchlist.items()}
        best_ticker, best_score = None, 0.0
        for ticker, name in names.items():
            score = difflib.SequenceMatcher(None, company_guess.lower(), name.lower()).ratio()
            if score > best_score:
                best_ticker, best_score = ticker, score
        if best_score >= fuzzy_threshold:
            return best_ticker

    return None
