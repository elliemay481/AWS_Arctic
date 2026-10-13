"""Search all collocation database files for one date."""
from pathlib import Path

import pandas as pd

CSV_DIR = Path("/home/behrensg/demo_jupyter/colocat_database/true/")
DATE = "2026-06-16"
TIME_COL = "Earthcare time"

for f in sorted(CSV_DIR.glob("*_colocations_ext.csv")):
    try:
        df = pd.read_csv(f)
    except Exception as err:
        print(f"{f.name}: could not read ({err})")
        continue
    if TIME_COL not in df.columns:
        print(f"{f.name}: no '{TIME_COL}' column")
        continue

    times = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    parsed_hit = times.dt.strftime("%Y-%m-%d") == DATE
    raw_hit = df[TIME_COL].astype(str).str.contains(DATE)
    rows = df[parsed_hit | raw_hit]
    if len(rows) == 0:
        continue

    lat_cols = [c for c in df.columns if c.startswith("col lat AWS swID 68")]
    print(f"\n{f.name}: {len(rows)} rows on {DATE} "
          f"({parsed_hit.sum()} parsed ok, {raw_hit.sum()} matched as text)")
    print(rows[[TIME_COL] + lat_cols].to_string())