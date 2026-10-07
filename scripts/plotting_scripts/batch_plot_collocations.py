"""Plot AWS L2 Arctic FWP or LWP on a map next to the collocated EarthCARE data.

Used to batch plot a large number of collocations.
Each (AWS, EarthCARE)
pair in --pairs-file whose AWS file is in --data-dir gets one figure:

    left column     map of the AWS FWP (or LWP) swath, with the EarthCARE track shown
    right column    AWS FWP and EarthCARE IWP along the track (or AWS LWP
                    and EarthCARE LWP), their relative difference, and the
                    EarthCARE IWC (or LWC) profiles, with the AWS and
                    EarthCARE mean mass height (Zm) on top for FWP

To plot for a single day, the --pattern argument can be used with,
e.g. *20251205*.nc

The pairs file lists the _v2 AWS files. To plot another version of the same
granules instead, give --version, e.g. --version v3: each _v2 file in the pairs
file is replaced by the file with the same name ending in _v3, and the figures
are saved with a _v3 suffix. A v3 file is only plotted if its _v2 version is
also in --data-dir; otherwise an error is printed and it is skipped.

Example
-------
python batch_plot_collocations.py \
    --data-dir /scratch/may/aws/L2_arctic/2025/12 \
    --pairs-file ../../data/earthcare_aws_file_pairs_2025-12_to_2026-02.pkl

python batch_plot_collocations.py \
    --data-dir /scratch/may/aws/L2_arctic/2025/12 \
    --pairs-file ../../data/earthcare_aws_file_pairs_2025-12_to_2026-02.pkl \
    --pattern *20251205*.nc \
    --variable lwp

python batch_plot_collocations.py \
    --data-dir /scratch/may/aws/L2_arctic/2025/12 \
    --pairs-file ../../data/earthcare_aws_file_pairs_2025-12_to_2026-02.pkl \
    --version v3
"""

import argparse
import glob
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
 
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean as cmc
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils


plt.style.use("../../plotstyling.mplstyle")

VAR_LAT = "latitude"
VAR_LON = "longitude"

PAIRS_VERSION = "v2"   # the AWS file version listed in the pairs files

# what is read and how it is labelled, for each --variable
#   aws_var       variable in the AWS L2 file, for the map
#   aws, ea, wc   keys returned by utils.colocate_pair_nearest_profile
#   aws_zm, ea_zm mean mass heights [m] drawn on the profile panel (None: not drawn)
VARIABLES = {
    "fwp": {
        "aws_var": "fwp_mean",
        "aws": "aws_fwp", "aws_q1": "aws_fwp_q1", "aws_q16": "aws_fwp_q16", "aws_q84": "aws_fwp_q84", "aws_q99": "aws_fwp_q99",
        "ea": "ea_iwp", "wc": "ea_iwc",
        "flags": ["median_distance_to_ea", "mean_distance_to_ea", "n_ea_profiles"],
        "aws_zm": "aws_zm", "ea_zm": "ea_zm",
        "name": "FWP", "map_name": "Fwp", "wc_name": "IWC",
        "cmap": cmc.cm.ice, "wc_vmin": 1e-6, "wc_vmax": 1e-3,
    },
    "lwp": {
        "aws_var": "lwp_mean",
        "aws": "aws_lwp", "aws_q1": "aws_lwp_q1", "aws_q16": "aws_lwp_q16", "aws_q84": "aws_lwp_q84", "aws_q99": "aws_lwp_q99",
        "ea": "ea_lwp", "wc": "ea_lwc",
        "aws_zm": None, "ea_zm": None,
        "name": "LWP", "map_name": "Lwp", "wc_name": "LWC",
        "cmap": cmc.cm.matter_r, "wc_vmin": 1e-6, "wc_vmax": 1e-3,
    },
}

PANEL_WIDTH = 7.0                  # right column
PANEL_HEIGHTS = (2, 2, 2)    # FWP, relative difference, IWC
PANEL_GAPS = (0.25, 0.65)          # below FWP, below relative difference
CBAR_HEIGHT = 0.18
IWC_CBAR_GAP = 0.8                 # room for the time labels under the IWC panel
MAP_CBAR_GAP = 0.3
COLUMN_GAP = 1.3                   # room for the right column's y labels
MARGINS = {"left": 0.2, "right": 0.3, "top": 1.1, "bottom": 0.7}


# matches the ..._20251205124638_20251205142250 part of a file name
DATETIME_PATTERN = re.compile(r"(\d{14})_(\d{14})")

# matches the version at the end of an AWS file name, e.g. the _v2 in ..._v2.nc
VERSION_PATTERN = re.compile(r"_(v\d+)\.nc$")


def format_datetime_range(stem):
    """Turn the timestamps in a file name into a readable title."""
    match = DATETIME_PATTERN.search(stem)
    if match is None:
        return stem
    fmt = "%Y%m%d%H%M%S"
    start = datetime.strptime(match.group(1), fmt)
    end = datetime.strptime(match.group(2), fmt)
    return f"{start:%Y-%m-%d %H:%M} to {end:%H:%M UTC}"


def with_version(name, version):
    """The same AWS file name with its version changed, e.g. ..._v2.nc -> ..._v3.nc."""
    return VERSION_PATTERN.sub(f"_{version}.nc", name)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True,
                   help="folder holding the L2 files")
    p.add_argument("--pairs-file", required=True,
                   help="file pairs saved by find_collocations.py")
    p.add_argument("--version", default=PAIRS_VERSION,
                   help="AWS file version to plot, e.g. v2 or v3 (default: v2, as in the pairs file)")
    p.add_argument("--variable", choices=sorted(VARIABLES), default="fwp",
                   help="plot AWS FWP against EarthCARE IWP, or AWS LWP against EarthCARE LWP")
    p.add_argument("--pattern", default="l2_arctic_*.nc",
                   help="glob pattern, relative to data-dir")
    p.add_argument("--fig-dir", default="../../figures/earthcare_collocation_examples",
                   help="where to write the figures")
    p.add_argument("--max-dist-km", type=float, default=15.0,
                   help="largest AWS pixel to EarthCARE profile distance")
    p.add_argument("--lat-min", type=float, default=60.0,
                   help="southern edge of the map")
    p.add_argument("--fwp-vmin", type=float, default=1e-2)
    p.add_argument("--fwp-vmax", type=float, default=1e0)
    p.add_argument("--lwp-vmin", type=float, default=1e-2)
    p.add_argument("--lwp-vmax", type=float, default=2e-1)
    p.add_argument("--marker-size", type=float, default=1.0,
                   help="scatter marker size; larger for coarser swaths")
    p.add_argument("--fit-extent", action="store_true",
                   help="fit the map to each swath instead of using lat-min")
    p.add_argument("--overwrite", action="store_true",
                   help="redo collocations whose figure already exists")
    args = p.parse_args()
    if not re.fullmatch(r"v\d+", args.version):
        p.error(f"--version should look like v2 or v3, not {args.version}")
    return args


def nearest_per_scan(col):
    """Indices of the closest collocated pixel in each AWS scan."""
    order = np.lexsort((col["dist_km"], col["scan"]))
    scan_sorted = col["scan"][order]
    keep = np.r_[True, scan_sorted[1:] != scan_sorted[:-1]]
    return order[keep]


def layout_axes(fig, ax_map, map_aspect):
    """Size the figure and place every axis, so the two columns line up.

    Returns the three right-hand panels and the two colour bar axes.
    """
    col_height = (sum(PANEL_HEIGHTS) + sum(PANEL_GAPS)
                  + IWC_CBAR_GAP + CBAR_HEIGHT)
    map_height = col_height - MAP_CBAR_GAP - CBAR_HEIGHT
    map_width = map_height * map_aspect

    fig_w = (MARGINS["left"] + map_width + COLUMN_GAP + PANEL_WIDTH
             + MARGINS["right"])
    fig_h = MARGINS["top"] + col_height + MARGINS["bottom"]
    fig.set_size_inches(fig_w, fig_h)

    def rect(x, top, w, h):
        """Axis position in figure fractions, from inches measured from the top left."""
        return [x / fig_w, 1 - (top + h) / fig_h, w / fig_w, h / fig_h]

    # left column: map, then its colour bar
    x = MARGINS["left"]
    top = MARGINS["top"]
    ax_map.set_position(rect(x, top, map_width, map_height))
    cax_map = fig.add_axes(rect(x + 0.1 * map_width, top + map_height + MAP_CBAR_GAP,
                                0.8 * map_width, CBAR_HEIGHT))

    # right column: three panels, then the IWC colour bar
    x = MARGINS["left"] + map_width + COLUMN_GAP
    panels = []
    for h, gap in zip(PANEL_HEIGHTS, (*PANEL_GAPS, IWC_CBAR_GAP)):
        panels.append(fig.add_axes(rect(x, top, PANEL_WIDTH, h)))
        top += h + gap
    cax_iwc = fig.add_axes(rect(x, top, PANEL_WIDTH, CBAR_HEIGHT))

    panels[1].sharex(panels[0])
    panels[2].sharex(panels[0])
    return panels, cax_map, cax_iwc


def style_time_axis(ax):
    """Time ticks and grid shared by the right-hand panels."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.minorticks_on()
    ax.grid(True, which="major", lw=0.8)
    ax.grid(True, which="minor", lw=0.3, alpha=0.5)


def plot_collocation(ds, col, title, out_path, args):

    """Map in the left column, collocation panels in the right column."""
    v = VARIABLES[args.variable]
    vmin = getattr(args, f"{args.variable}_vmin")
    vmax = getattr(args, f"{args.variable}_vmax")

    lat = ds[VAR_LAT].values
    lon = ds[VAR_LON].values
    aws_map = ds[v["aws_var"]].values

    sel = np.argsort(col["scan"], kind="stable")   # all points, in time order
    aws = col[v["aws"]][sel]
    ea = col[v["ea"]][sel]
    wc = col[v["wc"]][sel]
    x = mdates.date2num(col["scan"][sel])
    x_2d = np.broadcast_to(x[:, None], wc.shape)
    coloc_date = np.datetime_as_string(col["scan"][sel][0], unit="m").replace("T", " ")

    # the map comes first: its shape sets the size of the whole figure
    proj = ccrs.NorthPolarStereo() if args.lat_min >= 0 else ccrs.SouthPolarStereo()
    fig = plt.figure()
    ax_map = fig.add_axes([0, 0, 1, 1], projection=proj)
    if args.fit_extent:
        ax_map.set_extent([float(np.nanmin(lon)), float(np.nanmax(lon)),
                           float(np.nanmin(lat)), float(np.nanmax(lat))],
                          crs=ccrs.PlateCarree())
    else:
        ax_map.set_extent([-180, 180, args.lat_min, 90], crs=ccrs.PlateCarree())
    map_aspect = np.ptp(ax_map.get_xlim()) / np.ptp(ax_map.get_ylim())

    (ax_fwp, ax_cdf, ax_iwc), cax_map, cax_iwc = layout_axes(fig, ax_map, map_aspect)

    # ---- map ----
    ax_map.add_feature(cfeature.OCEAN, facecolor="lightgrey", zorder=0)
    ax_map.add_feature(cfeature.LAND, facecolor="dimgrey", zorder=0)
    ax_map.add_feature(cfeature.BORDERS, edgecolor="white", linewidth=0.5, zorder=1)
    ax_map.coastlines(color="white", linewidth=0.8, zorder=10)

    sc = ax_map.scatter(
        lon, lat, c=aws_map, s=args.marker_size,
        norm=LogNorm(vmin=vmin, vmax=vmax), cmap=v["cmap"],
        transform=ccrs.PlateCarree(),
        zorder=2, ec="none",
    )
    ax_map.scatter(col["lon"][sel], col["lat"][sel], c="red", s=15, ec="none",
                   transform=ccrs.PlateCarree(), zorder=3, label="EarthCARE")
    ax_map.gridlines(draw_labels=False, linewidth=0.3)
    ax_map.legend(loc="lower left", fontsize=12)
    ax_map.set_title(f"AWS {v['name']} swath with collocated EarthCARE track")
    fig.colorbar(sc, cax=cax_map, orientation="horizontal", extend="both",
                 label=rf"Retrieved {v['map_name']} (kg/m$^{{2}}$)")

    # label the ends of the track, matching the first and last times in the line plots
    track_lon = col["lon"][sel]
    track_lat = col["lat"][sel]
    geo = ccrs.PlateCarree()._as_mpl_transform(ax_map)
    for i, text, offset in [(0, "Start", (8, -8)), (-1, "End", (-8, 8))]:
        #ax_map.plot(track_lon[i], track_lat[i], marker="o", ms=6, color="black",
                    #transform=ccrs.PlateCarree(), zorder=4)
        ax_map.annotate(text, xy=(track_lon[i], track_lat[i]), xycoords=geo,
                        xytext=offset, textcoords="offset points",
                        ha="right" if offset[0] < 0 else "left",
                        fontsize=14, fontweight="bold", color="red", zorder=11,
                        )
                        #bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    # ---- FWP and IWP along the track ----
    ax = ax_fwp

    q16 = col[v["aws_q16"]][sel]
    q84 = col[v["aws_q84"]][sel]
    ax.fill_between(x, q16, q84, color="C0", alpha=0.2, lw=0, label="AWS Q16:Q84")
    ax.plot(x, aws, color="C0", label="AWS", lw=2)
    ax.plot(x, ea, color="C1", label="EarthCARE", lw=2)
    ax.set_yscale("log")
    ax.set_ylim([1e-2, 1e2])
    ax.axhspan(1e-2, 2e-1, color="grey", alpha=0.2)
    ax.set_ylabel(rf"{v['name']} [kg m$^{{-2}}$]")
    #ax.set_title(f"Collocation: {coloc_date} UTC")
    ax.legend(fontsize=12)
    style_time_axis(ax)
    ax.tick_params(labelbottom=False)

    # ---- relative difference ----
    """
    ax = ax_diff
    with np.errstate(divide="ignore", invalid="ignore"):
        diff_ea = 100 * (aws - ea) / ea
        diff_aws = 100 * (aws - ea) / aws
    ax.plot(x, diff_aws, color="black", ls=":", label="Wrt AWS", lw=2)
    ax.plot(x, diff_ea, color="C1", ls=":", label="Wrt EarthCARE", lw=2)
    ax.axhline(0, color="grey", ls="-", lw=2)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel(r"Relative difference [$\%$]")
    ax.legend()
    style_time_axis(ax)
    ax.tick_params(labelbottom=False)
    """
    # ---- cdf width ----
    q1 = col[v["aws_q1"]][sel]
    q99 = col[v["aws_q99"]][sel]
    ax = ax_cdf
    with np.errstate(divide="ignore", invalid="ignore"):
        width = (q99 - q1) / aws
        ax.fill_between(x, 0, width, where=aws > 1e-3,
                    color="C0", alpha=0.2, lw=0, label="AWS Unc.")
    
    #ax.plot(x, ea, color="C1", ls=":", label="EarthCARE", lw=2)
    ax.set_ylabel(r"(Q99 - Q1)/Mean")
    ax.legend(fontsize=12)
    style_time_axis(ax)
    ax.tick_params(labelbottom=False)
    
    # ---- collocation quality flags, on a second y-axis ----
    flag_labels = {
        "median_distance_to_ea": "Median distance to EC [km]",
        "mean_distance_to_ea": "Mean distance to EC [km]",
        "n_ea_profiles": "Number of EC profiles",
    }
    ax_flags = ax.twinx()
    for n, flag in enumerate(v.get("flags", [])[:2]):
        ax_flags.plot(x, col[flag][sel], color=f"C{n + 2}", lw=1.5,
                      label=flag_labels.get(flag, flag))
    ax_flags.set_ylabel("Flags [km, count]")
    ax_flags.set_ylim(bottom=0)
    ax_flags.tick_params(labelbottom=False)

    # one legend for both axes
    handles, labels = ax.get_legend_handles_labels()
    flag_handles, flag_labels_used = ax_flags.get_legend_handles_labels()
    ax.legend(handles + flag_handles, labels + flag_labels_used, fontsize=10, loc="upper left")

    # ---- EarthCARE IWC ----
    ax = ax_iwc
    pm = ax.pcolormesh(x_2d.T, col["ea_height"][sel].T / 1e3, wc.T,
                       shading="nearest", norm=LogNorm(vmin=v["wc_vmin"], vmax=v["wc_vmax"]),
                       cmap=v["cmap"])

    # mean mass height (Zm) of AWS and EarthCARE on top of the profiles, in km
    if v["aws_zm"] is not None:
        
        aws_zm = np.where(col["aws_fwp"][sel] < 1e-2, np.nan, col[v["aws_zm"]][sel])
        ax.plot(x, aws_zm / 1e3, color="C0", lw=2, label=r"AWS Z$_{m}$")
        ax.plot(x, col[v["ea_zm"]][sel] / 1e3, color="C1", lw=2, label=r"EarthCARE Z$_{m}$")
        ax.legend(loc="upper right", fontsize=12)

    ax.set_ylim(0, 15)
    ax.set_xlabel("AWS scan time")
    ax.set_ylabel("Height [km]")
    ax.set_title(f"EarthCARE {v['wc_name']} (nearest profile)")
    style_time_axis(ax)
    fig.colorbar(pm, cax=cax_iwc, orientation="horizontal", extend="both",
                 label=rf"{v['wc_name']} [kg m$^{{-3}}$]")

    fig.suptitle(f"QRNN model, {title}", y=1 - 0.3 / fig.get_figheight(),
                 va="top", fontsize=18)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    v = VARIABLES[args.variable]

    paths = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
    print(f"{len(paths)} files match {args.pattern} in {args.data_dir}")
    if not paths:
        return
    in_data_dir = {os.path.basename(p): p for p in paths}

    # the collocations whose AWS file, in the requested version, is one of the matched files.
    # The pairs file lists the _v2 files; for another version, the same name with that
    # version is used, but only if the _v2 file is in the data folder too.
    pairs = []
    for a, e in utils.load_file_pairs(args.pairs_file):
        pairs_name = os.path.basename(str(a))
        name = with_version(pairs_name, args.version)
        if name not in in_data_dir:
            continue
        if args.version != PAIRS_VERSION and pairs_name not in in_data_dir:
            print(f"error: {name} has no _{PAIRS_VERSION} version ({pairs_name}) "
                  f"in {args.data_dir}, not plotted")
            continue
        pairs.append((in_data_dir[name], e))
    print(f"{len(pairs)} collocations in {args.pairs_file} for these files "
          f"(AWS version {args.version})")

    # figures of other versions get the version as a suffix
    suffix = "" if args.version == PAIRS_VERSION else f"_{args.version}"

    t0 = time.time()
    for i, (aws_path, ea_path) in enumerate(pairs, 1):
        stem = os.path.splitext(os.path.basename(aws_path))[0]
        ea_stem = os.path.splitext(os.path.basename(str(ea_path)))[0]
        fig_path = os.path.join(args.fig_dir,
                                f"{stem}_earthcare_{ea_stem}_{args.variable}{suffix}.png")

        if os.path.exists(fig_path) and not args.overwrite:
            print(f"[{i}/{len(pairs)}] {stem}: figure exists, skipping")
            continue

        print(f"[{i}/{len(pairs)}] {stem}", flush=True)
        try:
            # "with" closes the file again once the figure is saved
            with xr.open_dataset(aws_path) as ds:
                missing = [name for name in (VAR_LAT, VAR_LON, v["aws_var"]) if name not in ds]
                if missing:
                    print(f"    missing variables {missing}, skipping")
                    continue

                col = utils.colocate_pair_nearest_profile(aws_path, ea_path,
                                                          max_dist_km=args.max_dist_km)
                missing = [key for key in (v["aws"], v["ea"], v["wc"]) if key not in col]
                if missing:
                    print(f"    colocate_pair_nearest_profile does not return {missing}, skipping")
                    continue
                if col["scan"].size == 0:
                    print(f"    no AWS pixels within {args.max_dist_km} km of EarthCARE, skipping")
                    continue

                plot_collocation(ds, col, format_datetime_range(stem), fig_path, args)
        except Exception as err:      # one bad file should not stop the run
            print(f"    failed: {type(err).__name__}: {err}")
            continue

    print(f"done in {(time.time() - t0) / 60:.1f} min, figures in {args.fig_dir}")


if __name__ == "__main__":
    main()