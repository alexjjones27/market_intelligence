"""Tiny on-disk cache.

Network pulls (Yahoo bars, SEC company facts, Wikipedia revisions) are slow and
rate-limited, and a backtest gets re-run dozens of times while it is being
built. Everything fetched lands in parquet/json under ``data/cache`` keyed by a
stable name, so a rebuild is cheap and reproducible.

The cache is intentionally dumb: no TTL, no invalidation. Historical daily bars
and filed XBRL facts do not change. Delete the directory to force a refetch.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)


class Cache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str, suffix: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_")
        p = self.root / f"{safe}{suffix}"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def frame(self, key: str, producer: Callable[[], pd.DataFrame], refresh: bool = False) -> pd.DataFrame:
        """Memoise a DataFrame to parquet."""
        path = self._path(key, ".parquet")
        if path.exists() and not refresh:
            try:
                return pd.read_parquet(path)
            except Exception as exc:  # pragma: no cover - corrupt cache
                logger.warning("cache read failed for %s (%s); refetching", key, exc)
        df = producer()
        try:
            df.to_parquet(path)
        except Exception as exc:  # pragma: no cover
            logger.warning("cache write failed for %s: %s", key, exc)
        return df

    def json(self, key: str, producer: Callable[[], Any], refresh: bool = False) -> Any:
        path = self._path(key, ".json")
        if path.exists() and not refresh:
            try:
                return json.loads(path.read_text())
            except Exception as exc:  # pragma: no cover
                logger.warning("cache read failed for %s (%s); refetching", key, exc)
        obj = producer()
        try:
            path.write_text(json.dumps(obj))
        except Exception as exc:  # pragma: no cover
            logger.warning("cache write failed for %s: %s", key, exc)
        return obj

    def has(self, key: str, suffix: str = ".parquet") -> bool:
        return self._path(key, suffix).exists()
