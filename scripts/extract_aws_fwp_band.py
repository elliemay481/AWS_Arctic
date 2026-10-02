"""Extract the collocated cases where AWS FWP falls in a narrow range.

Used to look at the band of cases where AWS retrieves FWP between about
2e-3 and 3e-3 kg m-2 whatever EarthCARE sees. The collocations are matched
exactly as in compute_aws_earthcare_coloc_stats.py (AWS pixel, all EarthCARE
profiles within footprint_radius_km), so the cases saved here are the ones
that make up that band in the histograms.

For each case it saves:
    AWS:        scan time, lat, lon, FWP, Zm, LWP, surface type fractions,
                file name, scan and fov index
    EarthCARE:  IWP, LWP and Zm averaged over the footprint, the mean IWC
                profile and its heights, and the number of profiles averaged

Reads a list of collocations created by find_collocations.py.
Mirrors the calling of find_collocations.py, i.e. calling with the argument 2025-07
loads the list of collocations created by calling 'python find_collocations.py 2025-07'

To use:
python extract_aws_fwp_band.py 2025-07                    # one month
python extract_aws_fwp_band.py 2025-12:2026-02            # range, both months included
python extract_aws_fwp_band.py 2025-07 2025-08 2026-06    # separate months
python extract_aws_fwp_band.py 2025-07:2025-08 2026-06    # ranges and months mixed
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from utils import load_earthcare

# ============================================================
# CONFIG
# ============================================================
fwp_min = 2e-3               # AWS FWP band to extract [kg m-2], fwp_min <= FWP < fwp_max
fwp_max = 3e-3
footprint_radius_km = 15.0   # same as compute_aws_earthcare_coloc_stats.py
min_ec_points = 1
km_per_deg = 111.2

AWS_VARS = {"fwp": "fwp_mean", "zm": "fwp_zm_mean", "lwp": "lwp_mean"}

data_dir = Path("/home/maye/AWS_Arctic/data")


# ============================================================
# HELPERS
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


def earthcare_zm(height, iwc):
    """Zm [m] of each profile: sum(z * iwc * dz) / sum(iwc * dz). NaN without ice."""
    dz = np.abs(np.gradient(height, axis=1))
    valid = np.isfinite(height) & np.isfinite(dz) & np.isfinite(iwc)
    z = np.where(valid, height, 0.0)
    ice_mass = np.where(valid, iwc * dz, 0.0)
    iwp = ice_mass.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        zm = (z * ice_mass).sum(axis=1) / iwp
    zm[iwp <= 0] = np.nan
    return zm, iwp


def nanmean_quiet(values, axis=None):
    """np.nanmean without the warning for all-NaN slices (they give NaN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


# ============================================================
# COLLOCATION PAIRS
# ============================================================
months, span = parse_months()

pairs_file = data_dir / f"earthcare_aws_file_pairs_{span}.pkl"
out_file = data_dir / f"aws_fwp_band_{fwp_min:g}_{fwp_max:g}_cases_{span}.pkl"

with open(pairs_file, "rb") as f:
    file_pairs = pickle.load(f)

print(f"{len(file_pairs)} file pairs")

surface_types = None   # names of the surface types, read from the first AWS file

# one list per saved field; each case appends one entry to each
cases = {key: [] for key in [
    "aws_file", "earthcare_file", "scan_index", "fov_index", "time", "lat", "lon",
    "aws_fwp", "aws_zm", "aws_lwp", "aws_surface_fractions", "aws_surface",
    "ea_iwp", "ea_lwp", "ea_zm", "ea_iwc", "ea_lwc", "ea_height", "n_ec_profiles",
]}

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
    aws = {key: ds[var][start:end, :].values.ravel() for key, var in AWS_VARS.items()}
    shape = ds.longitude[start:end, :].shape               # (scan, fov)
    scan_times = ds["scan"].values[start:end]

    # surface type fractions, one row per pixel
    frac_da = ds.surface_type_fractions.transpose("scan", "fov", "surface_type")
    names = [str(s) for s in frac_da.surface_type.values]
    frac = frac_da[start:end, :, :].values.reshape(-1, len(names))
    ds.close()

    if surface_types is None:
        surface_types = names
    elif names != surface_types:
        raise ValueError(f"surface types differ in {aws_filepath}: {names}")

    # EarthCARE Zm for every profile
    ea_zm, _ = earthcare_zm(ea["height"], ea["iwc"])

    # restrict EarthCARE to a box containing the AWS swath
    in_box = (
        (ea["lon"] >= np.nanmin(aws_lon))
        & (ea["lon"] <= np.nanmax(aws_lon))
        & (ea["lat"] >= np.nanmin(aws_lat))
        & (ea["lat"] <= np.nanmax(aws_lat))
    )
    box_idx = np.flatnonzero(in_box)       # EarthCARE profile index of each box entry
    if box_idx.size == 0:
        continue
    ea_lon_b = ea["lon"][box_idx]
    ea_lat_b = ea["lat"][box_idx]

    # AWS pixels in the FWP band, close in latitude to the EarthCARE track
    lat_margin = footprint_radius_km / km_per_deg
    candidates = np.flatnonzero(
        (aws_lat >= np.nanmin(ea_lat_b) - lat_margin)
        & (aws_lat <= np.nanmax(ea_lat_b) + lat_margin)
        & (aws["fwp"] >= fwp_min)
        & (aws["fwp"] < fwp_max)
    )

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

        # EarthCARE profiles in the footprint, with a valid IWP (as in the statistics)
        profiles = box_idx[near][inside]
        profiles = profiles[np.isfinite(ea["iwp"][profiles])]
        if profiles.size < min_ec_points:
            continue

        iwp = ea["iwp"][profiles]
        zm = ea_zm[profiles]
        has_ice = np.isfinite(zm) & (iwp > 0)

        scan_i, fov_i = np.unravel_index(j, shape)
        cases["aws_file"].append(Path(aws_filepath).name)
        cases["earthcare_file"].append(Path(ea_filepath).name)
        cases["scan_index"].append(start + scan_i)
        cases["fov_index"].append(fov_i)
        cases["time"].append(scan_times[scan_i])
        cases["lat"].append(aws_lat[j])
        cases["lon"].append(aws_lon[j])
        for key in AWS_VARS:
            cases[f"aws_{key}"].append(aws[key][j])

        # all fractions, and the name of the surface with the largest one
        # ("none" if the fractions are missing)
        cases["aws_surface_fractions"].append(frac[j])
        valid = np.isfinite(frac[j])
        cases["aws_surface"].append(surface_types[np.nanargmax(frac[j])] if valid.any() else "none")

        cases["ea_iwp"].append(iwp.mean())                          # zeros included
        cases["ea_lwp"].append(nanmean_quiet(ea["lwp"][profiles]))
        cases["ea_zm"].append(np.sum(zm[has_ice] * iwp[has_ice]) / np.sum(iwp[has_ice])
                              if has_ice.any() else np.nan)         # IWP-weighted
        cases["ea_iwc"].append(nanmean_quiet(ea["iwc"][profiles], axis=0))
        cases["ea_lwc"].append(nanmean_quiet(ea["lwc"][profiles], axis=0))
        cases["ea_height"].append(nanmean_quiet(ea["height"][profiles], axis=0))
        cases["n_ec_profiles"].append(profiles.size)

# ============================================================
# SAVE
# ============================================================
# scalars become 1D arrays (one entry per case); the IWC and height profiles
# become 2D arrays (case, level), and the surface fractions (case, surface type),
# with the columns in the order of output["surface_types"]
output = {key: np.array(values) for key, values in cases.items()}
output["surface_types"] = surface_types
output["config"] = {
    "fwp_min": fwp_min,
    "fwp_max": fwp_max,
    "footprint_radius_km": footprint_radius_km,
    "min_ec_points": min_ec_points,
}

print(f"\n{len(cases['time'])} cases with {fwp_min:g} <= AWS FWP < {fwp_max:g}")
for name in (surface_types or []) + ["none"]:
    n = int(np.sum(output["aws_surface"] == name)) if len(cases["time"]) else 0
    print(f"  {name:>8s}: {n}")

with open(out_file, "wb") as f:
    pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved to {out_file}")