"""Comparison of AWS and EarthCARE mean mass height (Zm) within bins.
Computes 1D histograms of all cases, for AWS Zm vs EarthCARE Zm.
Plus 2D histogram, i.e. AWS vs EarthCARE for each bin.
Also the 1D histogram of AWS mean mass diameter (Dm) at the same pixels;
EarthCARE has no Dm, so there is no EarthCARE counterpart.

The collocations are made with utils.colocate_pair_nearest_profile, the same
function as used for the collocation plots and the water path statistics:
one AWS pixel per fov, the one closest to the EarthCARE track (within
max_dist_km), with EarthCARE averaged over all profiles within avg_dist_km
of that pixel.

AWS Zm is the retrieved fwp_zm_mean. EarthCARE has no Zm, so utils calculates
it from the ice water content profiles as the IWC-weighted mean height:

    zm = sum(z * iwc * dz) / iwp,    with iwp = sum(iwc * dz)

Zm is only meaningful where there is enough ice, so only cases where both
the AWS FWP and the EarthCARE IWP are at least min_wp are used. Dm is
filtered on the AWS FWP in the same way.

Reads a list of collocations created by find_collocations.py.
Mirrors the calling of find_collocations.py, i.e. calling with the argument 2025-07
loads the list of collocations created by calling 'python find_collocations.py 2025-07'

To use:
python compute_aws_earthcare_zm_stats.py 2025-07                    # one month
python compute_aws_earthcare_zm_stats.py 2025-12:2026-02            # range, both months included
python compute_aws_earthcare_zm_stats.py 2025-07 2025-08 2026-06    # separate months
python compute_aws_earthcare_zm_stats.py 2025-07:2025-08 2026-06    # ranges and months mixed
python compute_aws_earthcare_zm_stats.py 2025-07:2026-06 --version v3

The pairs file lists the _v2 AWS files. To use another version of the same
granules instead, give --version, e.g. --version v3: each _v2 file in the pairs
file is replaced by the file with the same name ending in _v3, in the same
folder, and the output is saved with a _v3 suffix. A v3 file is only used if
its _v2 version exists too; otherwise an error is printed and it is skipped.

python compute_aws_earthcare_zm_stats.py 2025-12:2026-02 --version v3
"""

import argparse
import re
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

import utils

# ============================================================
# CONFIG
# ============================================================
max_dist_km = 15.0          # largest distance from the AWS pixel to the EarthCARE track
avg_dist_km = 15.0          # EarthCARE profiles within this distance of the pixel are averaged
min_ec_points = 1           # minimum EarthCARE profiles needed in that average
min_wp = 1e-2               # minimum AWS FWP and EarthCARE IWP [kg m-2] for Zm to be used
start, end = 0, 400         # AWS scans used from each file
zm_bins = np.linspace(0, 12000, 61)   # Zm bins [m], 200 m wide
dm_bins = np.linspace(0, 0.0015, 50)  # Dm bins [m]

data_dir = Path("/home/maye/AWS_Arctic/data")

PAIRS_VERSION = "v2"   # the AWS file version listed in the pairs files

# matches the version at the end of an AWS file name, e.g. the _v2 in ..._v2.nc
VERSION_PATTERN = re.compile(r"_(v\d+)\.nc$")


def with_version(path, version):
    """The same AWS file path with its version changed, e.g. ..._v2.nc -> ..._v3.nc."""
    path = Path(path)
    return path.with_name(VERSION_PATTERN.sub(f"_{version}.nc", path.name))


# ============================================================
# COLLOCATION PAIRS
# ============================================================
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


months, span, version = parse_months()

# output of other versions gets the version as a suffix
suffix = "" if version == PAIRS_VERSION else f"_{version}"

pairs_file = data_dir / f"earthcare_aws_file_pairs_{span}.pkl"
out_file = data_dir / f"aws_earthcare_zm_histograms_{span}{suffix}.pkl"

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

# AWS Dm, 1D only
dm_counts = np.zeros(len(dm_bins) - 1, dtype=np.int64)
dm_n = 0

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

    # pairs with enough EarthCARE profiles, enough ice in both, and a valid Zm in both
    valid = (
        (col["n_ea_profiles"] >= min_ec_points)
        & (col["aws_fwp"] >= min_wp)
        & (col["ea_iwp"] >= min_wp)
        & np.isfinite(col["aws_zm"])
        & np.isfinite(col["ea_zm"])
    )
    aws_m = col["aws_zm"][valid]
    ea_m = col["ea_zm"][valid]

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

    # AWS Dm at the collocated pixels with enough ice according to AWS
    dm_valid = (
        (col["n_ea_profiles"] >= min_ec_points)
        & (col["aws_fwp"] >= min_wp)
        & np.isfinite(col["aws_dm"])
    )
    dm_c, _ = np.histogram(col["aws_dm"][dm_valid], bins=dm_bins)
    dm_counts += dm_c
    dm_n += dm_c.sum()

bin_widths = np.diff(zm_bins)
dm_bin_widths = np.diff(dm_bins)

config = {
    "max_dist_km": max_dist_km,
    "avg_dist_km": avg_dist_km,
    "min_ec_points": min_ec_points,
    "min_wp": min_wp,
    "scans": (start, end),
    "aws_version": version,
}

# same layout as the water path statistics, under the keys "zm" and "dm"
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
        "config": config,
    },
    # AWS only: there is no EarthCARE Dm
    "dm": {
        "bins": dm_bins,
        "aws_counts": dm_counts,
        "aws_n": dm_n,
        "aws_hist_density": dm_counts / (dm_n * dm_bin_widths),
        "config": config,
    },
}

print(f"Zm: {aws_n} collocations used")
print(f"Dm: {dm_n} collocations used")

with open(out_file, "wb") as f:
    pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved to {out_file}")