"""
Find SLSTR (Cloud CCI / ORAC) granules that match AWS Arctic L2 files in time
and space.

Matching steps:
  1. Date: only look in SLSTR day-folders that fall within the AWS coverage.
  2. Location: skip SLSTR granules whose max latitude is below LAT_MIN.
  3. Time (file level): keep pairs where the SLSTR granule time range overlaps
     the AWS file time range, extended by +/- MAX_DT on each side.
  4. Content: skip SLSTR granules with no daylight ice pixels.
  5. Overlap (pixel level): only AWS footprints observed within +/- MAX_DT of
     the SLSTR granule are used. These and the SLSTR ice pixels are put onto a
     coarse lat/lon grid (GRID_RES degrees), and pairs are kept if they share
     at least MIN_SHARED_CELLS grid cells.

An SLSTR granule only spans about 3 minutes, so its pixels all have nearly the
same time; comparing each AWS footprint's scan time with the granule time
range is therefore effectively a per-pixel time check.

Output: a CSV with one row per matching (SLSTR, AWS) file pair.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
import re

import numpy as np
import pandas as pd
import xarray as xr

CLOUD_DIR = Path("/data/s6/L2_ext/CDS_C3S_CLOUD/2026/04/")
AWS_DIR = Path("/scratch/may/aws/L2_arctic/2026/04/")
OUT_CSV = "collocations_slstr_aws.csv"

LAT_MIN = 60.0
MAX_DT = timedelta(hours=1)

# spatial overlap check
GRID_RES = 0.5          # grid cell size in degrees
MIN_SHARED_CELLS = 1    # minimum number of shared grid cells to keep a pair

# Set to a single date to only check that SLSTR day-folder,
# or to None to check all dates covered by the AWS files.
ONLY_DATE = date(2026, 4, 25)

# Julian Day 2451545.0 is 2000-01-01 12:00 UTC
JD_J2000 = 2451545.0
J2000 = datetime(2000, 1, 1, 12)


def jd_to_datetime(jd):
    return J2000 + timedelta(days=float(jd) - JD_J2000)


def cell_keys(lat, lon, res=GRID_RES):
    """One number per point identifying its res-degree lat/lon grid cell (NaN if invalid)."""
    return np.floor(lat / res) * 10000 + np.floor(lon / res)


# AWS footprint grid cells and scan times, read once per file and reused
aws_cache = {}


def load_aws(f):
    if f not in aws_cache:
        with xr.open_dataset(f) as d:
            keys = cell_keys(d["latitude"].values, d["longitude"].values).ravel()
            # scan time for every footprint
            times = d["scan"].broadcast_like(d["latitude"]).values.ravel()
        ok = np.isfinite(keys)
        aws_cache[f] = (keys[ok], times[ok])
    return aws_cache[f]


# ---- 1. AWS files: start and end time from the filename ----
aws_pattern = re.compile(r"l2_arctic_(\d{14})_(\d{14})")
aws = []
for f in sorted(AWS_DIR.glob("l2_arctic_*.nc")):
    m = aws_pattern.search(f.name)
    if m is None:
        continue
    start = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    end = datetime.strptime(m.group(2), "%Y%m%d%H%M%S")
    aws.append((f, start, end))

if not aws:
    raise SystemExit(f"No AWS files found in {AWS_DIR.resolve()}")
print(f"{len(aws)} AWS files, {aws[0][1]} to {aws[-1][2]}")

# ---- 2. Dates to check in the SLSTR folders ----
# One extra day before each AWS file, because the SLSTR folder/filename date
# is the start of a processing segment, which can be hours before the
# granule's actual observation time.
dates = set()
for _, start, end in aws:
    d = (start - MAX_DT - timedelta(days=1)).date()
    while d <= (end + MAX_DT).date():
        dates.add(d)
        d += timedelta(days=1)

if ONLY_DATE is not None:
    if ONLY_DATE not in dates:
        print(f"Note: no AWS file is within +/- {MAX_DT} of {ONLY_DATE}, "
              "so no matches are expected")
    dates = {ONLY_DATE}
    print(f"Only checking SLSTR folder for {ONLY_DATE}")

cloud_files = []
for d in sorted(dates):
    day_dir = CLOUD_DIR / f"{d:%Y/%m/%d}"
    cloud_files += sorted(day_dir.glob("*/*.primary.nc"))
print(f"{len(cloud_files)} SLSTR files in matching date folders")

# ---- 3. Latitude filter, time matching, ice check, spatial overlap ----
rows = []
n_arctic = 0
n_no_ice = 0
n_time_pairs = 0
n_no_overlap = 0
for i, f in enumerate(cloud_files, 1):
    if i % 50 == 0:
        print(f"  checked {i}/{len(cloud_files)} files")
    try:
        # decode_times=False: the time units (Julian Days) do not decode
        # correctly, so we convert them by hand below
        with xr.open_dataset(f, decode_times=False) as ds:
            # location: skip granules that do not reach the Arctic
            lat_max = float(ds["lat"].max())
            if lat_max < LAT_MIN:
                continue
            n_arctic += 1

            # time: find AWS files within +/- MAX_DT
            t0 = jd_to_datetime(float(ds["time"].min()))
            t1 = jd_to_datetime(float(ds["time"].max()))
            matches = [(af, a0, a1) for af, a0, a1 in aws
                       if t0 <= a1 + MAX_DT and t1 >= a0 - MAX_DT]
            if not matches:
                continue

            # content: daylight ice-cloud pixels with a valid water path
            # (only checked for time matches, since it reads more data)
            iwp = ds["cwp"].where((ds["phase"] == 2) & (ds["illum"] == 1))
            ice = iwp.notnull().values
            n_ice = int(ice.sum())
            if n_ice == 0:
                n_no_ice += 1
                continue

            # grid cells covered by the ice pixels
            cci_keys = np.unique(cell_keys(ds["lat"].values[ice], ds["lon"].values[ice]))
    except Exception as e:
        print(f"  could not read {f.name}: {e}")
        continue

    # time window for AWS footprints, and granule mid time
    win_lo = np.datetime64(t0 - MAX_DT)
    win_hi = np.datetime64(t1 + MAX_DT)
    t_mid = np.datetime64(t0 + (t1 - t0) / 2)

    for af, a0, a1 in matches:
        n_time_pairs += 1
        try:
            a_keys, a_times = load_aws(af)
        except Exception as e:
            print(f"  could not read {af.name}: {e}")
            continue

        # only AWS footprints observed close in time to the granule
        in_time = (a_times >= win_lo) & (a_times <= win_hi)
        shared = np.intersect1d(cci_keys, a_keys[in_time])
        if shared.size < MIN_SHARED_CELLS:
            n_no_overlap += 1
            continue

        # time differences of the AWS footprints in the shared cells
        sel = in_time & np.isin(a_keys, shared)
        dt = np.abs((a_times[sel] - t_mid) / np.timedelta64(1, "m"))

        # 0 if the time ranges overlap, otherwise the gap between them
        gap = max(timedelta(0), a0 - t1, t0 - a1)
        rows.append({
            "cloud_file": str(f),
            "aws_file": str(af),
            "cloud_start": t0,
            "cloud_end": t1,
            "aws_start": a0,
            "aws_end": a1,
            "cloud_lat_max": round(lat_max, 2),
            "n_ice_pixels": n_ice,
            "n_shared_cells": int(shared.size),
            "n_shared_footprints": int(sel.sum()),
            "dt_median_minutes": round(float(np.median(dt)), 1),
            "dt_max_minutes": round(float(dt.max()), 1),
            "gap_minutes": round(gap.total_seconds() / 60, 1),
        })

print(f"{n_arctic} SLSTR granules reach north of {LAT_MIN} deg")
print(f"{n_no_ice} time-matched granules skipped: no daylight ice pixels")
print(f"{n_no_overlap} of {n_time_pairs} time-matched pairs skipped: "
      f"fewer than {MIN_SHARED_CELLS} shared {GRID_RES} deg grid cells "
      f"within +/- {MAX_DT}")

columns = ["cloud_file", "aws_file", "cloud_start", "cloud_end",
           "aws_start", "aws_end", "cloud_lat_max", "n_ice_pixels",
           "n_shared_cells", "n_shared_footprints",
           "dt_median_minutes", "dt_max_minutes", "gap_minutes"]
out = pd.DataFrame(rows, columns=columns)
out.to_csv(OUT_CSV, index=False)
print(f"{len(out)} matching pairs written to {OUT_CSV}")