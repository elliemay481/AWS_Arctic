"""Plot AWS L2 Arctic FWP or LWP on a map next to the collocated EarthCARE data.

Used to batch plot a large number of collocations.
Each (AWS, EarthCARE)
pair in --pairs-file whose AWS file is in --data-dir gets one figure:

    left column     map of the AWS FWP (or LWP) swath, with the EarthCARE track shown
    right column    AWS FWP and EarthCARE IWP along the track (or AWS LWP
                    and EarthCARE LWP), their relative difference, and the
                    EarthCARE IWC (or LWC) profiles

To plot for a single day, the --pattern argument can be used with,
e.g. *20251205*.nc

Example
-------
python batch_plot_collocations.py \
    --data-dir /scratch/may/aws/L2_arctic/2025/12 \
    --pairs-file ../data/earthcare_aws_file_pairs_2025-12_to_2026-02.pkl

python batch_plot_collocations.py \
    --data-dir /scratch/may/aws/L2_arctic/2025/12 \
    --pairs-file ../data/earthcare_aws_file_pairs_2025-12_to_2026-02.pkl \
    --pattern *20251205*.nc \
    --variable lwp
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

# what is read and how it is labelled, for each --variable
#   aws_var       variable in the AWS L2 file, for the map
#   aws, ea, wc   keys returned by utils.colocate_pair_nearest_profile
VARIABLES = {
    "fwp": {
        "aws_var": "fwp_mean",
        "aws": "aws_fwp", "ea": "ea_iwp", "wc": "ea_iwc",
        "name": "FWP", "map_name": "Fwp", "wc_name": "IWC",
        "cmap": cmc.cm.ice, "wc_vmin": 1e-6, "wc_vmax": 1e-3,
    },
    "lwp": {
        "aws_var": "lwp_mean",
        "aws": "aws_lwp", "ea": "ea_lwp", "wc": "ea_lwc",
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


def format_datetime_range(stem):
    """Turn the timestamps in a file name into a readable title."""
    match = DATETIME_PATTERN.search(stem)
    if match is None:
        return stem
    fmt = "%Y%m%d%H%M%S"
    start = datetime.strptime(match.group(1), fmt)
    end = datetime.strptime(match.group(2), fmt)
    return f"{start:%Y-%m-%d %H:%M} to {end:%H:%M UTC}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True,
                   help="folder holding the L2 files")
    p.add_argument("--pairs-file", required=True,
                   help="file pairs saved by find_collocations.py")
    p.add_argument("--variable", choices=sorted(VARIABLES), default="fwp",
                   help="plot AWS FWP against EarthCARE IWP, or AWS LWP against EarthCARE LWP")
    p.add_argument("--pattern", default="l2_arctic_*.nc",
                   help="glob pattern, relative to data-dir")
    p.add_argument("--fig-dir", default="../../figures/earthcare_collocation_examples",
                   help="where to write the figures")
    p.add_argument("--max-dist-km", type=float, default=5.0,
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
    return p.parse_args()


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

    sel = nearest_per_scan(col)
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

    (ax_fwp, ax_diff, ax_iwc), cax_map, cax_iwc = layout_axes(fig, ax_map, map_aspect)

    # ---- map ----
    ax_map.add_feature(cfeature.OCEAN, facecolor="lightgrey", zorder=0)
    ax_map.add_feature(cfeature.LAND, facecolor="dimgrey", zorder=0)
    ax_map.add_feature(cfeature.BORDERS, edgecolor="white", linewidth=0.5, zorder=1)
    ax_map.coastlines(color="white", linewidth=1, zorder=100)

    sc = ax_map.scatter(
        lon, lat, c=aws_map, s=args.marker_size,
        norm=LogNorm(vmin=vmin, vmax=vmax), cmap=v["cmap"],
        transform=ccrs.PlateCarree(),
        zorder=2, ec="none",
    )
    ax_map.scatter(col["lon"][sel], col["lat"][sel], c="red", s=15, ec="none",
                   transform=ccrs.PlateCarree(), zorder=3, label="EarthCARE")
    ax_map.gridlines(draw_labels=False, linewidth=0.3)
    ax_map.legend(loc="lower left")
    ax_map.set_title(f"AWS {v['name']} swath with collocated EarthCARE track")
    fig.colorbar(sc, cax=cax_map, orientation="horizontal", extend="both",
                 label=rf"Retrieved {v['map_name']} (kg/m$^{{2}}$)")

    # ---- FWP and IWP along the track ----
    ax = ax_fwp
    ax.plot(x, aws, color="black", label="AWS", lw=2)
    ax.plot(x, ea, color="C1", label="EarthCARE", lw=2)
    ax.set_yscale("log")
    ax.set_ylim([1e-3, 1e2])
    ax.set_ylabel(rf"{v['name']} [kg m$^{{-2}}$]")
    #ax.set_title(f"Collocation: {coloc_date} UTC")
    ax.legend()
    style_time_axis(ax)
    ax.tick_params(labelbottom=False)

    # ---- relative difference ----
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

    # ---- EarthCARE IWC ----
    ax = ax_iwc
    pm = ax.pcolormesh(x_2d, col["ea_height"][sel] / 1e3, wc,
                       shading="nearest", norm=LogNorm(vmin=v["wc_vmin"], vmax=v["wc_vmax"]),
                       cmap=v["cmap"])
    ax.set_ylim(0, 10)
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

    # the collocations whose AWS file is one of the matched files
    aws_by_name = {os.path.basename(p): p for p in paths}
    pairs = [(aws_by_name[os.path.basename(str(a))], e)
             for a, e in utils.load_file_pairs(args.pairs_file)
             if os.path.basename(str(a)) in aws_by_name]
    print(f"{len(pairs)} collocations in {args.pairs_file} for these files")

    t0 = time.time()
    for i, (aws_path, ea_path) in enumerate(pairs, 1):
        stem = os.path.splitext(os.path.basename(aws_path))[0]
        ea_stem = os.path.splitext(os.path.basename(str(ea_path)))[0]
        fig_path = os.path.join(args.fig_dir,
                                f"{stem}_earthcare_{ea_stem}_{args.variable}.png")

        if os.path.exists(fig_path) and not args.overwrite:
            print(f"[{i}/{len(pairs)}] {stem}: figure exists, skipping")
            continue

        print(f"[{i}/{len(pairs)}] {stem}", flush=True)
        try:
            ds = xr.open_dataset(aws_path)
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