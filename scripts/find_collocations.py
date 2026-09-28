"""Pair AWS L2 Arctic files with EarthCARE ACM_CAP_2B files at collocations.

Reads the collocation times from the database, finds the AWS and EarthCARE filepaths,
and saves a list of unique (aws_l2_arctic_file, earthcare_file) pairs.

Requires aws_processing repo to run.

To use:
pixi run python ../AWS_Arctic/scripts/find_collocations.py 2025-07              # one month
pixi run python ../AWS_Arctic/scripts/find_collocations.py 2025-12 2026-02      # range, both months included
"""

import argparse
import pickle
import re
from pathlib import Path

import pandas as pd

import colocation_proc as coloc


CSV_DIR = Path("/home/behrensg/demo_jupyter/colocat_database/true/")
AWS_ROOT = Path("/scratch/may/aws/L2_arctic")
OUTPUT_DIR = Path("/home/maye/AWS_Arctic/data/")

EARTHCARE_TIME_DELAY = 1.0

MAX_AWS_DURATION = pd.Timedelta(hours=24)  # removes a few strange looking AWS files

# variable names in collocation database
TIME_COL = "Earthcare time"
LAT_COL = "col lat AWS swID 68 [°N]"
CORR_COL = "corr coeff AWS33 vs. AWS44 bright. temp"

MONTH_NAMES = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "june",
    7: "july", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec",
}


def parse_months():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("start", help="first month, YYYY-MM")
    p.add_argument("end", nargs="?", help="last month, YYYY-MM (default: same as start)")
    args = p.parse_args()

    try:
        months = pd.period_range(args.start, args.end or args.start, freq="M")
    except ValueError as err:
        p.error(f"could not read the months: {err}")
    if len(months) == 0:
        p.error("the end month is before the start month")
    return months


def find_csv_files(year, month):
    """Database files for one month.
    Most are named <month>_<year>_..., but some 2025 files
    (e.g. july_..., oct_...) have no year in the name.
    """
    name = MONTH_NAMES[month]
    files = sorted(CSV_DIR.glob(f"{name}_{year}_*_colocations_ext.csv"))
    if files or year != 2025:
        return files
    return [f for f in sorted(CSV_DIR.glob(f"{name}_*_colocations_ext.csv"))
            if not re.fullmatch(r"\d{4}", f.name.split("_")[1])]


def index_aws_files(months):
    """(start, end, path) for every AWS file in the given months.

    The month before is included too, since a file that starts at the end
    of one month can cover collocations early in the next month.
    """
    index = []
    for period in [months[0] - 1, *months]:
        folder = AWS_ROOT / f"{period.year:04d}" / f"{period.month:02d}"
        for f in sorted(folder.rglob("l2_arctic_*_v2.nc")):
            try:
                start, end = (pd.to_datetime(t, format="%Y%m%d%H%M%S")
                              for t in f.stem.split("_")[2:4])
            except ValueError:
                print(f"  could not parse times from {f.name}, skipping")
                continue
            if end - start <= MAX_AWS_DURATION:
                index.append((start, end, f))
    return index


def find_aws_file(time, aws_index):
    """The first AWS file whose time span contains the given time."""
    return next((f for start, end, f in aws_index if start <= time <= end), None)


def find_earthcare_file(time):
    """The EarthCARE file for a collocation."""
    try:
        _, filename = coloc.earthcare_load(time, EARTHCARE_TIME_DELAY, "Arctic",
                                           product_type="/L2b/ACM_CAP_2B/")
    except Exception as err:      # one bad lookup should not stop the run
        print(f"    EarthCARE lookup failed at {time}: {type(err).__name__}: {err}")
        return None
    return filename


def read_times(csv_file, year, month):
    try:
        df = pd.read_csv(csv_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as err:
        print(f"    could not read {csv_file.name} ({err}), skipping")
        return []

    needed = [TIME_COL, LAT_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"    {csv_file.name} is missing columns {missing}, skipping")
        return []

    df = df[df[LAT_COL] > 0]                           # Arctic only

    # to be able to compare with the AWS file times:
    times = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    times = times.dt.tz_localize(None).dropna()
    return times[(times.dt.year == year) & (times.dt.month == month)]


def process_month(period, aws_index):
    """(aws_file, earthcare_file) for each collocation in one month."""
    print(f"\n===== {period} =====", flush=True)
    csv_files = find_csv_files(period.year, period.month)
    print(f"  {len(csv_files)} database files")

    times = []
    for csv_file in csv_files:
        #print(f"  reading {csv_file.name}", flush=True)
        times.extend(read_times(csv_file, period.year, period.month))

    pairs = []
    n_no_aws = n_no_ec = 0
    for time in sorted(times):
        aws_file = find_aws_file(time, aws_index)
        if aws_file is None:
            n_no_aws += 1
            continue
        earthcare_file = find_earthcare_file(time)
        if earthcare_file is None:
            n_no_ec += 1
            continue
        pairs.append((aws_file, earthcare_file))

    print(f"  {len(times)} collocations: {n_no_aws} without AWS file, "
          f"{n_no_ec} without EarthCARE file, {len(pairs)} file pairs")
    return pairs


def main():

    from collections import Counter
    
    year, month = 2026, 3
    counts = Counter()
    for f in find_csv_files(year, month):
        counts.update(set(read_times(f, year, month)))
    
    repeated = sum(1 for n in counts.values() if n > 1)
    print(f"{len(counts)} unique times, {repeated} appear in more than one file")
    
    months = parse_months()
    aws_index = index_aws_files(months)
    print(f"Processing {len(months)} months, {months[0]} to {months[-1]}, "
          f"{len(aws_index)} AWS files indexed")

    pairs = []
    for period in months:
        pairs += process_month(period, aws_index)

    # many collocations share the same two files; keep each pair once, in order
    pairs = list(dict.fromkeys(pairs))

    span = f"{months[0]}" if len(months) == 1 else f"{months[0]}_to_{months[-1]}"
    output_file = OUTPUT_DIR / f"earthcare_aws_file_pairs_{span}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved {len(pairs)} file pairs to {output_file}")


if __name__ == "__main__":
    main()