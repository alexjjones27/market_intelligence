"""Fetch and cache the full market panel (prices + PIT fundamentals + macro)."""
import logging, sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
from alpha_lab.config import load_config, Config
from alpha_lab.data.panel import build_panel

cfg = load_config() if Path("configs/default.yaml").exists() else Config()
panel = build_panel(cfg)
print("\n=== PANEL SUMMARY ===")
print("dates:", panel.dates.min().date(), "->", panel.dates.max().date(), f"({len(panel.dates)})")
print("tickers:", len(panel.tickers))
print("fields:", sorted(panel.fields))
print("mean tradeable/day:", round(panel.diagnostics["mean_tradeable_per_day"], 1))
print("price failures:", len(panel.diagnostics["price_failures"]))
print("\nfundamental coverage:")
print(panel.diagnostics["fundamental_coverage"].to_string(index=False))
print("\nuniverse coverage (tail):")
print(panel.diagnostics["universe_coverage"][["snapshot_date","n_members","n_with_prices","pct_missing"]].tail(6).to_string(index=False))
Path("data/cache").mkdir(parents=True, exist_ok=True)
with open("data/cache/panel.pkl","wb") as fh: pickle.dump(panel, fh)
print("\nwrote data/cache/panel.pkl")
