"""Fetch and cache point-in-time S&P 500 membership snapshots."""
import logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
from alpha_lab.config import load_config, Config
from alpha_lab.data.universe import load_universe

cfg = load_config() if Path("configs/default.yaml").exists() else Config()
u = load_universe(cfg)
print("snapshots:", len(u.snapshot_dates), u.snapshot_dates[0].date(), "->", u.snapshot_dates[-1].date())
print("distinct tickers ever:", len(u.tickers))
print("members per snapshot:", u.membership.sum(axis=1).describe().to_dict())
print("sectors:", u.sectors.value_counts().to_dict())
print("cik coverage:", int(u.ciks.notna().sum()), "/", len(u.ciks))
