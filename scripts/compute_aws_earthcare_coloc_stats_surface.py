"""Comparison of AWS and EarthCARE water paths within bins, for FWP and LWP,
split by surface type.
Computes 1D histograms of all cases, for AWS FWP vs EarthCARE IWP and
AWS LWP vs EarthCARE LWP, for each surface type.
Plus 2D histogram, i.e. AWS vs EarthCARE for each bin, for each surface type.

Each AWS pixel is given the surface type with the largest fraction in
surface_type_fractions. Pixels with fraction below min_dominant_fraction are excluded.

Reads a list of collocations created by find_collocations.py.
Mirrors the calling of find_collocations.py, i.e. calling with the argument 2025-07
loads the list of collocations created by calling 'python find_collocations.py 2025-07'

To use:
python compute_aws_earthcare_coloc_stats_surface.py 2025-07                    # one month
python compute_aws_earthcare_coloc_stats_surface.py 2025-12:2026-02            # range, both months included
python compute_aws_earthcare_coloc_stats_surface.py 2025-07 2025-08 2026-06    # separate months
python compute_aws_earthcare_coloc_stats_surface.py 2025-07:2025-08 2026-06    # ranges and months mixed
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
min_dominant_fraction = 0.8
km_per_deg = 111.2
wp_bins = np.concatenate([[0.0], np.logspace(-4, 2, 100)])

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
out_file = data_dir / f"aws_earthcare_histograms_surface_{span}.pkl"

with open(pairs_file, "rb") as f:
    file_pairs = pickle.load(f)

print(f"{len(file_pairs)} file pairs")

# ============================================================
# STATS (created after the first file, once the surface types are known)
# ============================================================
n_bins = len(wp_bins) - 1
surface_types = None
stats = None

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

    # surface type fractions, one row per pixel
    frac_da = ds.surface_type_fractions.transpose("scan", "fov", "surface_type")
    names = [str(s) for s in frac_da.surface_type.values]
    frac = frac_da[start:end, :, :].values.reshape(-1, len(names))
    ds.close()

    if surface_types is None:
        surface_types = names
        n_surf = len(surface_types)
        stats = {
            q: {
                "aws_counts": np.zeros((n_surf, n_bins), dtype=np.int64),
                "ea_counts": np.zeros((n_surf, n_bins), dtype=np.int64),
                # 2D joint histogram: rows = EarthCARE bins, columns = AWS bins
                "joint_counts": np.zeros((n_surf, n_bins, n_bins), dtype=np.int64),
                "aws_n": np.zeros(n_surf, dtype=np.int64),
                "ea_n": np.zeros(n_surf, dtype=np.int64),
                "n_unclassified": 0,
            }
            for q in QUANTITIES
        }
    elif names != surface_types:
        raise ValueError(f"surface types differ in {aws_filepath}: {names}")

    # dominant surface type per pixel (-1 = unclassified)
    frac_filled = np.where(np.isfinite(frac), frac, -1.0)
    aws_surf = np.argmax(frac_filled, axis=1)
    dominant_frac = frac_filled[np.arange(frac.shape[0]), aws_surf]
    aws_surf[dominant_frac <= 0] = -1
    aws_surf[dominant_frac < min_dominant_fraction] = -1

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
    surf_coloc = {q: [] for q in QUANTITIES}
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
            surf_coloc[q].append(aws_surf[j])

    for q, st in stats.items():
        aws_m = np.array(aws_coloc[q])
        ea_m = np.array(ea_coloc[q])
        surf_m = np.array(surf_coloc[q], dtype=int)

        st["n_unclassified"] += int(np.sum(surf_m < 0))

        # histograms per surface type
        for i_s in range(len(surface_types)):
            on_surf = surf_m == i_s
            if not on_surf.any():
                continue

            # 1D histogram counts
            aws_c, _ = np.histogram(aws_m[on_surf], bins=wp_bins)
            ea_c, _ = np.histogram(ea_m[on_surf], bins=wp_bins)

            st["aws_counts"][i_s] += aws_c
            st["ea_counts"][i_s] += ea_c

            # 2D joint histogram
            jc, _, _ = np.histogram2d(ea_m[on_surf], aws_m[on_surf], bins=[wp_bins, wp_bins])
            st["joint_counts"][i_s] += jc.astype(np.int64)

            # number of samples that contributed to histogram
            st["aws_n"][i_s] += aws_c.sum()
            st["ea_n"][i_s] += ea_c.sum()

bin_widths = np.diff(wp_bins)

# one entry per quantity, e.g. output["fwp"]["aws_counts"][i] for surface_types[i]
#   counts:       aws_counts, ea_counts    shape (n_surface, n_bins)
#   joint counts: joint_counts             shape (n_surface, n_bins, n_bins)
#                 rows = EarthCARE bins, columns = AWS bins
#   totals:       aws_n, ea_n              shape (n_surface,)
with np.errstate(divide="ignore", invalid="ignore"):   # surfaces with no samples give NaN
    output = {
        q: {
            "bins": wp_bins,
            "surface_types": surface_types,
            **st,
            "aws_hist_density": st["aws_counts"] / (st["aws_n"][:, None] * bin_widths),
            "ea_hist_density": st["ea_counts"] / (st["ea_n"][:, None] * bin_widths),
            "config": {
                "footprint_radius_km": footprint_radius_km,
                "min_ec_points": min_ec_points,
                "min_dominant_fraction": min_dominant_fraction,
            },
        }
        for q, st in stats.items()
    }

for q, o in output.items():
    print(f"\n{q}:")
    for name, n in zip(o["surface_types"], o["aws_n"]):
        print(f"  {name:>8s}: {n}")
    print(f"  unclassified: {o['n_unclassified']}")

with open(out_file, "wb") as f:
    pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\nSaved to {out_file}")