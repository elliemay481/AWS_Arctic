"""Comparison of AWS and EarthCARE mean mass height (Zm) within bins.
Computes 1D histograms of all cases, for AWS Zm vs EarthCARE Zm.
Plus 2D histogram, i.e. AWS vs EarthCARE for each bin.

AWS Zm is the retrieved fwp_zm. EarthCARE has no Zm, so it is calculated
from each ice water content profile as the IWC-weighted mean height:

    zm = sum(z * iwc * dz) / iwp,    with iwp = sum(iwc * dz)

Zm is only meaningful where there is enough ice, so only cases where both
the AWS FWP and the EarthCARE IWP are at least min_wp are used.

Reads a list of collocations created by find_collocations.py.
Mirrors the calling of find_collocations.py, i.e. calling with the argument 2025-07
loads the list of collocations created by calling 'python find_collocations.py 2025-07'

To use:
python compute_aws_earthcare_zm_stats.py 2025-07                    # one month
python compute_aws_earthcare_zm_stats.py 2025-12:2026-02            # range, both months included
python compute_aws_earthcare_zm_stats.py 2025-07 2025-08 2026-06    # separate months
python compute_aws_earthcare_zm_stats.py 2025-07:2025-08 2026-06    # ranges and months mixed
"""

import argparse
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
min_ec_points = 1           # minimum EarthCARE profiles needed inside the distance
min_wp = 1e-2               # minimum AWS FWP and EarthCARE IWP [kg m-2] for Zm to be used
km_per_deg = 111.2
zm_bins = np.linspace(0, 12000, 61)   # Zm bins [m], 200 m wide

AWS_ZM = "fwp_zm_mean"   # AWS Zm [m]
AWS_FWP = "fwp_mean"   # AWS FWP, for the min_wp cut

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


def earthcare_zm(ea):
    """Zm [m] and IWP [kg m-2] of each EarthCARE profile, from the IWC profile.

    zm  = sum(z * iwc * dz) / iwp
    iwp = sum(iwc * dz)

    z is the height of each grid centre and dz the thickness of each layer.
    Missing IWC counts as no ice. Profiles without ice get Zm = NaN.
    """
    height = ea["height"]    # (profile, level) [m]
    iwc = ea["iwc"]          # (profile, level) [kg m-3]

    # layer thickness around each grid centre, from the spacing of the centres;
    # abs() because the heights may run from the top down
    dz = np.abs(np.gradient(height, axis=1))

    # levels with a missing height, thickness or IWC contribute nothing
    valid = np.isfinite(height) & np.isfinite(dz) & np.isfinite(iwc)
    z = np.where(valid, height, 0.0)
    ice_mass = np.where(valid, iwc * dz, 0.0)   # ice per layer [kg m-2]

    iwp = ice_mass.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        zm = (z * ice_mass).sum(axis=1) / iwp
    zm[iwp <= 0] = np.nan
    return zm, iwp


months, span = parse_months()

pairs_file = data_dir / f"earthcare_aws_file_pairs_{span}.pkl"
out_file = data_dir / f"aws_earthcare_zm_histograms_{span}.pkl"

with open(pairs_file, "rb") as f:
    file_pairs = pickle.load(f)

print(f"{len(file_pairs)} file pairs")

# ============================================================
# STATS
# ============================================================
n_bins = len(zm_bins) - 1
aws_counts = np.zeros(n_bins, dtype=np.int64)
ea_counts = np.zeros(n_bins, dtype=np.int64)
# 2D joint histogram: rows = EarthCARE bins, columns = AWS bins
joint_counts = np.zeros((n_bins, n_bins), dtype=np.int64)
aws_n = 0
ea_n = 0

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
    aws_zm = ds[AWS_ZM][start:end, :].values.ravel()
    aws_fwp = ds[AWS_FWP][start:end, :].values.ravel()
    ds.close()

    # EarthCARE Zm and IWP for every profile
    ea_zm, ea_iwp = earthcare_zm(ea)

    # restrict EarthCARE to a box containing the AWS swath
    in_box = (
        (ea["lon"] >= np.nanmin(aws_lon))
        & (ea["lon"] <= np.nanmax(aws_lon))
        & (ea["lat"] >= np.nanmin(aws_lat))
        & (ea["lat"] <= np.nanmax(aws_lat))
    )
    ea_lon_b = ea["lon"][in_box]
    ea_lat_b = ea["lat"][in_box]
    ea_zm_b = ea_zm[in_box]
    ea_iwp_b = ea_iwp[in_box]

    if ea_lon_b.size == 0:
        continue

    # only check AWS pixels whose latitude is close to the EarthCARE track,
    # and that have a valid Zm and enough ice
    lat_margin = footprint_radius_km / km_per_deg
    candidates = np.flatnonzero(
        (aws_lat >= np.nanmin(ea_lat_b) - lat_margin)
        & (aws_lat <= np.nanmax(ea_lat_b) + lat_margin)
        & np.isfinite(aws_zm)
        & (aws_fwp >= min_wp)
    )

    # loop over AWS pixels: combine all EarthCARE profiles inside the footprint
    aws_coloc = []
    ea_coloc = []
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

        zm = ea_zm_b[near][inside]
        iwp = ea_iwp_b[near][inside]
        ok = np.isfinite(iwp)
        if ok.sum() < min_ec_points:
            continue

        # mean IWP over the footprint, clear profiles included
        iwp_mean = iwp[ok].mean()
        if iwp_mean < min_wp:
            continue

        # Zm of the footprint: the IWP-weighted mean of the profiles' Zm, which is
        # the same as the Zm of the footprint's mean IWC profile
        has_ice = ok & np.isfinite(zm) & (iwp > 0)
        zm_footprint = np.sum(zm[has_ice] * iwp[has_ice]) / np.sum(iwp[has_ice])

        aws_coloc.append(aws_zm[j])
        ea_coloc.append(zm_footprint)

    aws_m = np.array(aws_coloc)
    ea_m = np.array(ea_coloc)

    # 1D histogram counts
    aws_c, _ = np.histogram(aws_m, bins=zm_bins)
    ea_c, _ = np.histogram(ea_m, bins=zm_bins)

    aws_counts += aws_c
    ea_counts += ea_c

    # 2D joint histogram
    jc, _, _ = np.histogram2d(ea_m, aws_m, bins=[zm_bins, zm_bins])
    joint_counts += jc.astype(np.int64)

    # number of samples that contributed to histogram
    aws_n += aws_c.sum()
    ea_n += ea_c.sum()

bin_widths = np.diff(zm_bins)

# same layout as the water path statistics, under the key "zm"
output = {
    "zm": {
        "bins": zm_bins,
        "aws_counts": aws_counts,
        "ea_counts": ea_counts,
        "joint_counts": joint_counts,
        "aws_n": aws_n,
        "ea_n": ea_n,
        "aws_hist_density": aws_counts / (aws_n * bin_widths),
        "ea_hist_density": ea_counts / (ea_n * bin_widths),
        "config": {
            "footprint_radius_km": footprint_radius_km,
            "min_ec_points": min_ec_points,
            "min_wp": min_wp,
        },
    }
}

print(f"{aws_n} collocations used")

with open(out_file, "wb") as f:
    pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved to {out_file}")