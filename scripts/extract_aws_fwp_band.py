"""Extract the collocated cases where AWS FWP falls in a narrow range.

Used to look at the band of cases where AWS retrieves FWP between about
2e-3 and 3e-3 kg m-2 whatever EarthCARE sees. The collocations are made with
utils.colocate_pair_nearest_profile, the same function as used for the
statistics and plots (one AWS pixel per fov, EarthCARE averaged within
avg_dist_km), so the cases saved here are the ones that make up that band
in the histograms.

For each case it saves:
    AWS:        scan time, lat, lon, FWP, Zm, LWP, surface type fractions,
                file name, scan and fov index, distance to EarthCARE
    EarthCARE:  IWP, LWP and Zm, the IWC and LWC profiles and their heights
                (all as returned by the collocation function), and the
                number of profiles averaged

Reads a list of collocations created by find_collocations.py.
Mirrors the calling of find_collocations.py, i.e. calling with the argument 2025-07
loads the list of collocations created by calling 'python find_collocations.py 2025-07'

To use:
python extract_aws_fwp_band.py 2025-07 --version v3                   # one month
python extract_aws_fwp_band.py 2025-12:2026-02            # range, both months included
python extract_aws_fwp_band.py 2025-07 2025-08 2026-06    # separate months
python extract_aws_fwp_band.py 2025-07:2025-08 2026-06    # ranges and months mixed

The pairs file lists the _v2 AWS files. To use another version of the same
granules instead, give --version, e.g. --version v3: each _v2 file in the pairs
file is replaced by the file with the same name ending in _v3, in the same
folder, and the output is saved with a _v3 suffix. A v3 file is only used if
its _v2 version exists too; otherwise an error is printed and it is skipped.

python extract_aws_fwp_band.py 2025-12:2026-02 --version v3
"""

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import utils

# ============================================================
# CONFIG
# ============================================================
fwp_min = 2e-3              # AWS FWP band to extract [kg m-2], fwp_min <= FWP < fwp_max
fwp_max = 3e-3
max_dist_km = 15.0          # same as compute_aws_earthcare_coloc_stats.py
avg_dist_km = 15.0
min_ec_points = 1
start, end = 0, 400         # AWS scans used from each file

data_dir = Path("/home/maye/AWS_Arctic/data")

PAIRS_VERSION = "v2"   # the AWS file version listed in the pairs files

# matches the version at the end of an AWS file name, e.g. the _v2 in ..._v2.nc
VERSION_PATTERN = re.compile(r"_(v\d+)\.nc$")


# ============================================================
# HELPERS
# ============================================================
def with_version(path, version):
    """The same AWS file path with its version changed, e.g. ..._v2.nc -> ..._v3.nc."""
    path = Path(path)
    return path.with_name(VERSION_PATTERN.sub(f"_{version}.nc", path.name))


def parse_months():
    """The months to process, a name for them to use in file names, and the AWS version.

    Same as in find_collocations.py, so the same arguments give the same file name.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("months", nargs="+",
                   help="months as YYYY-MM, or ranges as YYYY-MM:YYYY-MM (both included)")
    p.add_argument("--version", default=PAIRS_VERSION,
                   help="AWS file version to use, e.g. v2 or v3 (default: v2, as in the pairs file)")
    args = p.parse_args()
    if not re.fullmatch(r"v\d+", args.version):
        p.error(f"--version should look like v2 or v3, not {args.version}")

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
    return sorted(set(months)), "_".join(names), args.version


# ============================================================
# COLLOCATION PAIRS
# ============================================================
months, span, version = parse_months()

# output of other versions gets the version as a suffix
suffix = "" if version == PAIRS_VERSION else f"_{version}"

pairs_file = data_dir / f"earthcare_aws_file_pairs_{span}.pkl"
out_file = data_dir / f"aws_fwp_band_{fwp_min:g}_{fwp_max:g}_cases_{span}{suffix}.pkl"

with open(pairs_file, "rb") as f:
    file_pairs = pickle.load(f)

print(f"{len(file_pairs)} file pairs")

# the pairs file lists the _v2 files; for another version, use the file with the same
# name in that version, but only if the _v2 file exists too
if version != PAIRS_VERSION:
    swapped = []
    for aws_filepath, ea_filepath in file_pairs:
        new_path = with_version(aws_filepath, version)
        if not Path(aws_filepath).exists():
            print(f"error: {new_path.name} has no _{PAIRS_VERSION} version "
                  f"({Path(aws_filepath).name}), not used")
            continue
        if not new_path.exists():
            print(f"    no {version} file {new_path.name}, skipping")
            continue
        swapped.append((new_path, ea_filepath))
    file_pairs = swapped
    print(f"{len(file_pairs)} file pairs with AWS version {version}")

surface_types = None   # names of the surface types, read from the first AWS file

# one list per saved field; each case appends one entry to each
cases = {key: [] for key in [
    "aws_file", "earthcare_file", "scan_index", "fov_index", "time", "lat", "lon",
    "aws_fwp", "aws_zm", "aws_lwp", "aws_surface_fractions", "aws_surface", "dist_km",
    "ea_iwp", "ea_iwp_closest", "ea_lwp", "ea_zm", "ea_iwc", "ea_lwc",
    "ea_ice_mass_flux", "ea_rwc", "ea_height", "n_ec_profiles",
]}

# ============================================================
# PROCESS EACH FILE PAIR
# ============================================================
n_pairs = len(file_pairs)
for k, (aws_filepath, ea_filepath) in enumerate(file_pairs, 1):
    print(f"[{k}/{n_pairs}] {Path(aws_filepath).name}", flush=True)

    try:
        col = utils.colocate_pair_nearest_profile(
            aws_filepath, ea_filepath, start=start, end=end,
            max_dist_km=max_dist_km, avg_dist_km=avg_dist_km,
        )
    except Exception as err:      # one bad file should not stop the run
        print(f"    failed: {type(err).__name__}: {err}")
        continue

    # collocations in the FWP band, with enough valid EarthCARE profiles (as in the statistics)
    in_band = np.flatnonzero(
        (col["aws_fwp"] >= fwp_min)
        & (col["aws_fwp"] < fwp_max)
        & (col["n_ea_profiles"] >= min_ec_points)
        & np.isfinite(col["ea_iwp"])
    )
    if in_band.size == 0:
        continue

    # surface type fractions and scan index of these pixels, from the AWS file.
    # The collocation gives each pixel's scan time and fov; the scan time is
    # looked up in the file to get its scan index.
    with xr.open_dataset(aws_filepath) as ds:
        names = [str(s) for s in ds.surface_type.values]
        scan_times = ds["scan"].values
        scan_idx = np.searchsorted(scan_times, col["scan"][in_band])
        fov_idx = col["fov"][in_band].astype(int)
        frac = (ds.surface_type_fractions.transpose("scan", "fov", "surface_type")
                .values[scan_idx, fov_idx, :])

    if surface_types is None:
        surface_types = names
    elif names != surface_types:
        raise ValueError(f"surface types differ in {aws_filepath}: {names}")

    for n, c in enumerate(in_band):
        cases["aws_file"].append(Path(aws_filepath).name)
        cases["earthcare_file"].append(Path(ea_filepath).name)
        cases["scan_index"].append(scan_idx[n])
        cases["fov_index"].append(fov_idx[n])
        cases["time"].append(col["scan"][c])
        cases["lat"].append(col["lat"][c])
        cases["lon"].append(col["lon"][c])
        cases["aws_fwp"].append(col["aws_fwp"][c])
        cases["aws_zm"].append(col["aws_zm"][c])
        cases["aws_lwp"].append(col["aws_lwp"][c])
        cases["dist_km"].append(col["dist_km"][c])

        # all fractions, and the name of the surface with the largest one
        # ("none" if the fractions are missing)
        cases["aws_surface_fractions"].append(frac[n])
        valid = np.isfinite(frac[n])
        cases["aws_surface"].append(surface_types[np.nanargmax(frac[n])] if valid.any() else "none")

        cases["ea_iwp"].append(col["ea_iwp"][c])
        cases["ea_iwp_closest"].append(col["ea_iwp_closest"][c])
        cases["ea_lwp"].append(col["ea_lwp"][c])
        cases["ea_zm"].append(col["ea_zm"][c])
        cases["ea_iwc"].append(col["ea_iwc"][c])
        cases["ea_lwc"].append(col["ea_lwc"][c])
        cases["ea_ice_mass_flux"].append(col["ea_ice_mass_flux"][c])
        cases["ea_rwc"].append(col["ea_rwc"][c])
        cases["ea_height"].append(col["ea_height"][c])
        cases["n_ec_profiles"].append(col["n_ea_profiles"][c])

# ============================================================
# SAVE
# ============================================================
# scalars become 1D arrays (one entry per case); the IWC, LWC and height profiles
# become 2D arrays (case, level), and the surface fractions (case, surface type),
# with the columns in the order of output["surface_types"]
output = {key: np.array(values) for key, values in cases.items()}
output["surface_types"] = surface_types
output["config"] = {
    "fwp_min": fwp_min,
    "fwp_max": fwp_max,
    "max_dist_km": max_dist_km,
    "avg_dist_km": avg_dist_km,
    "min_ec_points": min_ec_points,
    "scans": (start, end),
    "aws_version": version,
}

print(f"\n{len(cases['time'])} cases with {fwp_min:g} <= AWS FWP < {fwp_max:g}")
for name in (surface_types or []) + ["none"]:
    n = int(np.sum(output["aws_surface"] == name)) if len(cases["time"]) else 0
    print(f"  {name:>8s}: {n}")

with open(out_file, "wb") as f:
    pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved to {out_file}")