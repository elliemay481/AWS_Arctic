"""Comparison of AWS and EarthCARE water paths within bins, for FWP and LWP.
Computes 1D histograms of all cases, for AWS FWP vs EarthCARE IWP and
AWS LWP vs EarthCARE LWP.
Plus 2D histogram, i.e. AWS vs EarthCARE for each bin.

Reads a list of collocations created by find_collocations.py.
Mirrors the calling of find_collocations.py, i.e. calling with the argument 2025-07
loads the list of collocations created by calling 'python find_collocations.py 2025-07'

To use:
python compute_aws_earthcare_coloc_stats.py 2025-07                    # one month
python compute_aws_earthcare_coloc_stats.py 2025-12:2026-02            # range, both months included
python compute_aws_earthcare_coloc_stats.py 2025-07 2025-08 2026-06    # separate months
python compute_aws_earthcare_coloc_stats.py 2025-07:2025-08 2026-06    # ranges and months mixed
"""

import argparse
import sys
import numpy as np
import pandas as pd
import xarray as xr
import pickle
from pathlib import Path

from utils import load_earthcare

# ============================================================
# CONFIG
# ============================================================
footprint_radius_km = 15.0   # Include all EarthCARE obs within this distance from an AWS obs
min_ec_points = 1           # minimum EarthCARE points needed inside the distance
km_per_deg = 111.2
wp_bins = np.concatenate([[0.0], np.logspace(-4, 2, 100)])   # water path bins, used for FWP and LWP

# for each quantity: the AWS variable, and the EarthCARE key from load_earthcare
QUANTITIES = {
    "fwp": ("fwp_mean", "iwp"),
    "lwp": ("lwp_mean", "lwp"),
}

data_dir = Path("/home/maye/AWS_Arctic/data")

# ============================================================
# COLLOCATION PAIRS
# ============================================================
def parse_months():
    """The months to process, and a name for them to use in file names.

    Same as in find_collocations.py, so the same arguments give the same file name.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("months", nargs="+",
                   help="months as YYYY-MM, or ranges as YYYY-MM:YYYY-MM (both included)")
    args = p.parse_args()

    months, names = [], []
    for item in args.months:
        start, _, end = item.partition(":")
        try:
            rng = pd.period_range(start, end or start, freq="M")
        except ValueError as err:
            p.error(f"could not read {item}: {err}")
        if len(rng) == 0:
            p.error(f"{item}: the end month is before the start month")
        months.extend(rng)
        names.append(f"{rng[0]}" if len(rng) == 1 else f"{rng[0]}_to_{rng[-1]}")
    return sorted(set(months)), "_".join(names)


months, span = parse_months()

pairs_file = data_dir / f"earthcare_aws_file_pairs_{span}.pkl"
out_file = data_dir / f"aws_earthcare_histograms_{span}.pkl"

with open(pairs_file, "rb") as f:
    file_pairs = pickle.load(f)

print(f"{len(file_pairs)} file pairs")

# ============================================================
# STATS
# ============================================================
n_bins = len(wp_bins) - 1
stats = {
    q: {
        "aws_counts": np.zeros(n_bins, dtype=np.int64),
        "ea_counts": np.zeros(n_bins, dtype=np.int64),
        # 2D joint histogram: rows = EarthCARE bins, columns = AWS bins
        "joint_counts": np.zeros((n_bins, n_bins), dtype=np.int64),
        "aws_n": 0,
        "ea_n": 0,
    }
    for q in QUANTITIES
}

# ============================================================
# PROCESS EACH FILE PAIR
# ============================================================
n_pairs = len(file_pairs)
for k, (aws_filepath, ea_filepath) in enumerate(file_pairs, 1):
    print(f"[{k}/{n_pairs}] {Path(aws_filepath).name}", flush=True)
    ds = xr.open_dataset(aws_filepath)
    ea = load_earthcare(ea_filepath)

    start = 0
    end = 400

    # AWS pixels, flattened to 1D (one entry per pixel)
    aws_lon = ds.longitude[start:end, :].values.ravel()
    aws_lat = ds.latitude[start:end, :].values.ravel()
    aws_vals = {q: ds[aws_var][start:end, :].values.ravel()
                for q, (aws_var, _) in QUANTITIES.items()}
    ds.close()

    # restrict EarthCARE to a box containing the AWS swath
    in_box = (
        (ea["lon"] >= np.nanmin(aws_lon))
        & (ea["lon"] <= np.nanmax(aws_lon))
        & (ea["lat"] >= np.nanmin(aws_lat))
        & (ea["lat"] <= np.nanmax(aws_lat))
    )
    ea_lon_b = ea["lon"][in_box]
    ea_lat_b = ea["lat"][in_box]
    ea_vals_b = {q: ea[ea_key][in_box] for q, (_, ea_key) in QUANTITIES.items()}

    if ea_lon_b.size == 0:
        continue

    # only check AWS pixels whose latitude is close to the EarthCARE track,
    # and that have at least one valid AWS value
    lat_margin = footprint_radius_km / km_per_deg
    candidates = np.flatnonzero(
        (aws_lat >= np.nanmin(ea_lat_b) - lat_margin)
        & (aws_lat <= np.nanmax(ea_lat_b) + lat_margin)
        & np.logical_or.reduce([np.isfinite(v) for v in aws_vals.values()])
    )

    # loop over AWS pixels: average all EarthCARE points inside the footprint
    aws_coloc = {q: [] for q in QUANTITIES}
    ea_coloc = {q: [] for q in QUANTITIES}
    for j in candidates:
        # first cut by latitude
        dlat = ea_lat_b - aws_lat[j]
        near = np.abs(dlat) * km_per_deg <= footprint_radius_km
        if not near.any():
            continue

        # distance in km (wrap longitude around)
        dlon = ea_lon_b[near] - aws_lon[j]
        dlon = (dlon + 180.0) % 360.0 - 180.0
        dx = dlon * np.cos(np.deg2rad(aws_lat[j])) * km_per_deg
        dy = dlat[near] * km_per_deg
        inside = np.sqrt(dx * dx + dy * dy) <= footprint_radius_km

        for q in QUANTITIES:
            if not np.isfinite(aws_vals[q][j]):
                continue
            vals = ea_vals_b[q][near][inside]
            vals = vals[np.isfinite(vals)]
            if vals.size < min_ec_points:
                continue

            aws_coloc[q].append(aws_vals[q][j])
            ea_coloc[q].append(vals.mean())   # zeros included

    for q, s in stats.items():
        aws_m = np.array(aws_coloc[q])
        ea_m = np.array(ea_coloc[q])

        # 1D histogram counts
        aws_c, _ = np.histogram(aws_m, bins=wp_bins)
        ea_c, _ = np.histogram(ea_m, bins=wp_bins)

        s["aws_counts"] += aws_c
        s["ea_counts"] += ea_c

        # 2D joint histogram
        jc, _, _ = np.histogram2d(ea_m, aws_m, bins=[wp_bins, wp_bins])
        s["joint_counts"] += jc.astype(np.int64)

        # number of samples that contributed to histogram
        s["aws_n"] += aws_c.sum()
        s["ea_n"] += ea_c.sum()

bin_widths = np.diff(wp_bins)

# one entry per quantity, e.g. output["fwp"]["aws_counts"]
output = {
    q: {
        "bins": wp_bins,
        **s,
        "aws_hist_density": s["aws_counts"] / (s["aws_n"] * bin_widths),
        "ea_hist_density": s["ea_counts"] / (s["ea_n"] * bin_widths),
    }
    for q, s in stats.items()
}

with open(out_file, "wb") as f:
    pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)