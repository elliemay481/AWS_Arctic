"""Comparison of AWS and EarthCARE water paths within bins, for FWP and LWP.
Computes 1D histograms of all cases, for AWS FWP vs EarthCARE IWP and
AWS LWP vs EarthCARE LWP.
Plus 2D histogram, i.e. AWS vs EarthCARE for each bin.

The collocations are made with utils.colocate_pair_nearest_profile, the same
function as used for the collocation plots: one AWS pixel per fov, the one
closest to the EarthCARE track (within max_dist_km), with EarthCARE averaged
over all profiles within avg_dist_km of that pixel.

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
start, end = 0, 400         # AWS scans used from each file
wp_bins = np.concatenate([[0.0], np.logspace(-4, 2, 100)])   # water path bins, used for FWP and LWP

# for each quantity: the AWS and EarthCARE keys returned by utils.colocate_pair_nearest_profile
QUANTITIES = {
    "fwp": ("aws_fwp", "ea_iwp"),
    "lwp": ("aws_lwp", "ea_lwp"),
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

    try:
        col = utils.colocate_pair_nearest_profile(
            aws_filepath, ea_filepath, start=start, end=end,
            max_dist_km=max_dist_km, avg_dist_km=avg_dist_km,
        )
    except Exception as err:      # one bad file should not stop the run
        print(f"    failed: {type(err).__name__}: {err}")
        continue

    # enough EarthCARE profiles in the average
    enough_ec = col["n_ea_profiles"] >= min_ec_points

    for q, (aws_key, ea_key) in QUANTITIES.items():
        s = stats[q]

        # pairs where both values are valid for this quantity
        valid = enough_ec & np.isfinite(col[aws_key]) & np.isfinite(col[ea_key])
        aws_m = col[aws_key][valid]
        ea_m = col[ea_key][valid]

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
        "config": {
            "max_dist_km": max_dist_km,
            "avg_dist_km": avg_dist_km,
            "min_ec_points": min_ec_points,
            "scans": (start, end),
        },
    }
    for q, s in stats.items()
}

for q, s in stats.items():
    print(f"{q}: {s['aws_n']} collocations")

with open(out_file, "wb") as f:
    pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved to {out_file}")