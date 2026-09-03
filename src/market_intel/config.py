"""Central configuration. Loads from environment / .env.

Every provider-specific credential is optional. Absence of a key does not
raise; it flips the corresponding ingestor into MOCK mode. See
`ingestion/*.py` docstrings for what that means per source, and
.env.example for the full list of knobs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
EXPORTS_DIR = PROJECT_ROOT / "exports"


@dataclass(frozen=True)
class Settings:
    db_path: str = field(
        default_factory=lambda: os.environ.get(
            "MARKET_INTEL_DB_PATH", str(PROJECT_ROOT / "market_intel.db")
        )
    )
    sec_user_agent: str = field(
        default_factory=lambda: os.environ.get(
            "SEC_EDGAR_USER_AGENT", "Market Intel MVP (unset-contact@example.com)"
        )
    )
    finnhub_api_key: str | None = field(
        default_factory=lambda: os.environ.get("FINNHUB_API_KEY") or None
    )
    fred_api_key: str | None = field(
        default_factory=lambda: os.environ.get("FRED_API_KEY") or None
    )
    earnings_provider: str = field(
        default_factory=lambda: os.environ.get("EARNINGS_PROVIDER", "mock")
    )
    options_provider: str = field(
        default_factory=lambda: os.environ.get("OPTIONS_PROVIDER", "mock")
    )
    watchlist_path: str = field(
        default_factory=lambda: os.environ.get(
            "WATCHLIST_PATH", str(DATA_DIR / "watchlist.json")
        )
    )
    sector_map_path: str = field(
        default_factory=lambda: os.environ.get(
            "SECTOR_MAP_PATH", str(DATA_DIR / "sector_map.json")
        )
    )

    @property
    def news_is_mocked(self) -> bool:
        return not self.finnhub_api_key

    @property
    def macro_is_mocked(self) -> bool:
        return not self.fred_api_key

    @property
    def earnings_is_mocked(self) -> bool:
        return self.earnings_provider.lower() == "mock"

    @property
    def options_is_mocked(self) -> bool:
        return self.options_provider.lower() == "mock"


settings = Settings()
