"""Plot retrieved variables or antenna temperatures from L2 Arctic files.

Used to batch plot a large number of swaths. To plot for a single day, the
--pattern argument can be used with, e.g. *20260605*.nc

Which pair of retrieved variables is plotted is chosen with --variables:

    fwp_lwp    frozen and liquid water path
    zm_dm      mean mass height (zm) and mean mass diameter (dm)

What is plotted for them is chosen with --statistic:

    mean, median, most_prob
    relative_cdf_width        (q84 - q16) / mean, from <var>_quantiles and
                               <var>_mean

or with --channel, which plots antenna temperatures instead. Naming one channel also plots its
partner, as per:

    AWS21 + AWS31, AWS31 + AWS32, AWS33 + AWS44,
    AWS34 + AWS43, AWS35 + AWS42, AWS36 + AWS41


Example
-------
python batch_plot_l2.py \
    --data-dir /scratch/may/aws/L2_arctic/2025/12 \
    --fig-dir ../figures \
    --variables fwp_lwp \
    --statistic relative_cdf_width
    --pattern *20260605*.nc

python batch_plot_l2.py \
    --data-dir /scratch/may/aws/L2_arctic/2025/12 \
    --fig-dir ../figures \
    --variables zm_dm \
    --statistic median

python batch_plot_l2.py \
    --data-dir /scratch/may/aws/L2_arctic/2025/12 \
    --fig-dir ../figures \
    --channel AWS33
"""

import argparse
import glob
import os
import re
import time
from datetime import datetime

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean as cmc
import cmcrameri as cmcr


plt.style.use("../plotstyling.mplstyle")

VAR_LAT = "latitude"
VAR_LON = "longitude"
VAR_TB = "tb"
CHANNEL_DIM = "channel"
QUANTILE_DIM = "quantile"

Q_LOW, Q_HIGH = 0.16, 0.84
Q_MEDIAN = 0.5

STAT_LABELS = {
    "mean": "Posterior mean",
    "median": "Posterior median",
    "most_prob": "Posterior most probable",
    "relative_cdf_width": "Relative CDF width, (q84 - q16) / mean",
}

VARIABLE_SETS = {
    "fwp_lwp": ("fwp", "lwp"),
    "zm_dm": ("fwp_zm", "fwp_dm"),
}

VAR_STYLE = {
    "fwp": {"name": "Fwp", "units": r"kg/m$^{2}$", "cmap": cmc.cm.ice, "log": True},
    "lwp": {"name": "Lwp", "units": r"kg/m$^{2}$", "cmap": cmc.cm.matter_r, "log": True},
    "fwp_zm": {"name": "Zm", "units": None, "cmap": cmc.cm.thermal, "log": False},
    "fwp_dm": {"name": "Dm", "units": None, "cmap": cmc.cm.haline, "log": False},
}


# channels are plotted in pairs; naming either one plots both
CHANNEL_PAIRS = [
    ("AWS21", "AWS31"),
    ("AWS31", "AWS32"),
    ("AWS33", "AWS44"),
    ("AWS34", "AWS43"),
    ("AWS35", "AWS42"),
    ("AWS36", "AWS41"),
]

TB_VMIN, TB_VMAX = 100.0, 270.0


def channel_pair(name):
    """The pair a channel belongs to, in the order listed above.

    AWS31 appears in two pairs; the first match wins, so asking for AWS31
    gives AWS21 + AWS31. Ask for AWS32 to get AWS31 + AWS32.
    """
    for pair in CHANNEL_PAIRS:
        if name in pair:
            return pair
    known = sorted({c for pair in CHANNEL_PAIRS for c in pair})
    raise SystemExit(f"unknown channel {name}; known channels: {', '.join(known)}")


def quantile_field(ds, prefix, q):
    """One quantile of a retrieved variable, from the stored quantiles."""
    return ds[f"{prefix}_quantiles"].sel({QUANTILE_DIM: q}, method="nearest").values


def stat_field(ds, prefix, statistic):
    """The retrieved field for one variable and one statistic."""
    if statistic == "median":
        return quantile_field(ds, prefix, Q_MEDIAN)
    if statistic == "most_prob":
        return ds[f"{prefix}_most_prob"].values
    return ds[f"{prefix}_mean"].values


def stat_source_vars(statistic):
    """Which variables a file must hold for this statistic, per prefix."""
    if statistic == "median":
        return ["quantiles"]
    if statistic == "most_prob":
        return ["most_prob"]
    if statistic == "relative_cdf_width":
        return ["quantiles", "mean"]
    return ["mean"]


def relative_width(ds, prefix):
    """(q84 - q16) / mean for one variable, from the stored quantiles."""
    high = quantile_field(ds, prefix, Q_HIGH)
    low = quantile_field(ds, prefix, Q_LOW)
    mean = ds[f"{prefix}_mean"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (high - low) / mean
    out[~np.isfinite(out)] = np.nan
    return out


def auto_limits(arrays, low_pct=2.0, high_pct=98.0):
    """Colour limits from the percentiles of the finite, positive values."""
    values = np.concatenate([a[np.isfinite(a) & (a > 0)].ravel() for a in arrays])
    if values.size == 0:
        return 1e-2, 1e0
    vmin = float(np.percentile(values, low_pct))
    vmax = float(np.percentile(values, high_pct))
    if vmin <= 0 or vmax <= vmin:
        vmin, vmax = max(values.min(), 1e-6), values.max()
    return vmin, vmax


def field_norm(prefix, values, args):
    """Colour norm for one retrieved variable.

    Uses the --<prefix>-vmin/--<prefix>-vmax arguments where given and
    fills in any that are missing from the data.
    """
    vmin = getattr(args, f"{prefix}_vmin")
    vmax = getattr(args, f"{prefix}_vmax")
    if vmin is None or vmax is None:
        auto_min, auto_max = auto_limits([values])
        vmin = auto_min if vmin is None else vmin
        vmax = auto_max if vmax is None else vmax
    if VAR_STYLE[prefix]["log"]:
        return LogNorm(vmin=vmin, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def field_label(prefix):
    """Panel title for a retrieved variable."""
    style = VAR_STYLE[prefix]
    if style["units"] is None:
        return f"Retrieved {style['name']}"
    return f"Retrieved {style['name']} ({style['units']})"


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
    p.add_argument("--pattern", default="l2_arctic_*.nc",
                   help="glob pattern, relative to data-dir")
    p.add_argument("--fig-dir", default="../figures_for_me",
                   help="where to write the maps")
    p.add_argument("--variables", choices=sorted(VARIABLE_SETS),
                   default="fwp_lwp",
                   help="which pair of retrieved variables to plot")
    p.add_argument("--statistic",
                   choices=["mean", "median", "most_prob", "relative_cdf_width"],
                   default="mean",
                   help="which retrieved field to plot")
    p.add_argument("--channel", default=None,
                   help="plot antenna temperatures for this channel and its "
                        "partner, instead of a retrieved field")
    p.add_argument("--list-vars", action="store_true",
                   help="print the variables in the first file and stop")
    p.add_argument("--lat-min", type=float, default=60.0,
                   help="southern edge of the map")
    p.add_argument("--fwp-vmin", type=float, default=1e-2)
    p.add_argument("--fwp-vmax", type=float, default=1e0)
    p.add_argument("--lwp-vmin", type=float, default=1e-2)
    p.add_argument("--lwp-vmax", type=float, default=2e-1)
    p.add_argument("--zm-vmin", dest="fwp_zm_vmin", type=float, default=None,
                   help="default: 2nd percentile of each granule")
    p.add_argument("--zm-vmax", dest="fwp_zm_vmax", type=float, default=None,
                   help="default: 98th percentile of each granule")
    p.add_argument("--dm-vmin", dest="fwp_dm_vmin", type=float, default=None,
                   help="default: 2nd percentile of each granule")
    p.add_argument("--dm-vmax", dest="fwp_dm_vmax", type=float, default=None,
                   help="default: 98th percentile of each granule")
    p.add_argument("--marker-size", type=float, default=1.0,
                   help="scatter marker size; larger for coarser swaths")
    p.add_argument("--fit-extent", action="store_true",
                   help="fit the map to each swath instead of using lat-min")
    p.add_argument("--overwrite", action="store_true",
                   help="redo granules whose figure already exists")
    return p.parse_args()


def build_panels(ds, args):
    """The two panels to draw: (values, label, norm, cmap) for each."""
    if args.channel is not None:
        pair = channel_pair(args.channel)
        return [
            (ds[VAR_TB].sel({CHANNEL_DIM: name}).values,
             f"{name} $T_{{a}}$ (K)",
             Normalize(vmin=120, vmax=270),
             cmcr.cm.lajolla)
            for name in pair
        ]

    prefixes = VARIABLE_SETS[args.variables]

    if args.statistic == "relative_cdf_width":
        widths = [relative_width(ds, p) for p in prefixes]
        vmin, vmax = auto_limits(widths)
        return [
            (w, f"{VAR_STYLE[p]['name']} relative posterior width",
             Normalize(vmin=vmin, vmax=vmax), cmcr.cm.batlow)
            for p, w in zip(prefixes, widths)
        ]

    panels = []
    for p in prefixes:
        values = stat_field(ds, p, args.statistic)
        panels.append((values, field_label(p), field_norm(p, values, args),
                       VAR_STYLE[p]["cmap"]))
    return panels


def required_vars(args):
    """Variables a file must hold for the chosen plot."""
    if args.channel is not None:
        return [VAR_LAT, VAR_LON, VAR_TB]
    prefixes = VARIABLE_SETS[args.variables]
    suffixes = stat_source_vars(args.statistic)
    return [VAR_LAT, VAR_LON] + [f"{p}_{s}" for p in prefixes for s in suffixes]


def subtitle(args):
    """The line printed under the granule title."""
    if args.channel is not None:
        pair = channel_pair(args.channel)
        return f"$T_{{a}}$, {pair[0]} and {pair[1]}"
    return STAT_LABELS[args.statistic]


def plot_granule(ds, title, out_path, args):
    """Two-panel map for one granule."""
    lat = ds[VAR_LAT].values
    lon = ds[VAR_LON].values

    proj = ccrs.NorthPolarStereo() if args.lat_min >= 0 else ccrs.SouthPolarStereo()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             subplot_kw={"projection": proj})

    for ax, (values, label, norm, cmap) in zip(axes, build_panels(ds, args)):
        ax.add_feature(cfeature.OCEAN, facecolor="lightgrey", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="dimgrey", zorder=0)
        ax.add_feature(cfeature.BORDERS, edgecolor="white", linewidth=0.5, zorder=1)
        ax.coastlines(color="white", linewidth=1, zorder=100)

        sc = ax.scatter(
            lon, lat, c=values, s=args.marker_size,
            norm=norm, cmap=cmap,
            transform=ccrs.PlateCarree(),
            zorder=2, ec="none",
        )
        ax.gridlines(draw_labels=False, linewidth=0.3)
        ax.set_title(label)

        if args.fit_extent:
            ax.set_extent([float(np.nanmin(lon)), float(np.nanmax(lon)),
                           float(np.nanmin(lat)), float(np.nanmax(lat))],
                          crs=ccrs.PlateCarree())
        else:
            ax.set_extent([-180, 180, args.lat_min, 90], crs=ccrs.PlateCarree())

        fig.colorbar(sc, ax=ax, shrink=0.6, extend="both", pad=0.02)

    fig.suptitle(f"QRNN model, {title}\n{subtitle(args)}", y=0.99, fontsize=18)
    fig.subplots_adjust(wspace=0.02)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
    print(f"{len(paths)} files match {args.pattern} in {args.data_dir}")
    if not paths:
        return

    if args.list_vars:
        ds = xr.open_dataset(paths[0])
        print(ds)
        print(f"variables in {os.path.basename(paths[0])}:")
        for name in sorted(ds.data_vars):
            print(f"  {name}  {ds[name].dims}")
        if CHANNEL_DIM in ds.coords:
            print(f"channels: {list(ds[CHANNEL_DIM].values)}")
        if QUANTILE_DIM in ds.coords:
            print(f"quantiles: {list(ds[QUANTILE_DIM].values)}")
        return

    # a tag for the file name, so runs do not overwrite each other
    if args.channel is not None:
        pair = channel_pair(args.channel)
        tag = f"tb_{pair[0]}_{pair[1]}"
        print(f"plotting {subtitle(args)}")
    else:
        tag = f"{args.variables}_{args.statistic}"
        print(f"plotting {args.variables}, {subtitle(args)}")

    t0 = time.time()
    for i, path in enumerate(paths, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        fig_path = os.path.join(args.fig_dir, f"{stem}_{tag}.png")

        if os.path.exists(fig_path) and not args.overwrite:
            print(f"[{i}/{len(paths)}] {stem}: figure exists, skipping")
            continue

        print(f"[{i}/{len(paths)}] {stem}", flush=True)
        try:
            ds = xr.open_dataset(path)
            missing = [v for v in required_vars(args) if v not in ds]
            if missing:
                print(f"    missing variables {missing}, skipping "
                      f"(use --list-vars to see what the file holds)")
                continue

            plot_granule(ds, format_datetime_range(stem), fig_path, args)
        except Exception as err:      # one bad file should not stop the run
            print(f"    failed: {type(err).__name__}: {err}")
            continue

    print(f"done in {(time.time() - t0) / 60:.1f} min, figures in {args.fig_dir}")


if __name__ == "__main__":
    main()