"""Common ingestion interface. Every source module implements `fetch()`
and returns a list of `RawItem`. Nothing downstream cares which source
produced an item beyond its `source_tier` and `is_mocked` flag.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from market_intel.models.raw_item import RawItem


class BaseIngestor(ABC):
    source_tier: str = "secondary"
    is_mocked: bool = False

    @abstractmethod
    def fetch(self, tickers: list[str]) -> list[RawItem]:
        """Return raw items for the given tickers. Must not raise on a
        single bad ticker/network hiccup for one ticker -- log and skip,
        keep going for the rest of the watchlist."""
        raise NotImplementedError
