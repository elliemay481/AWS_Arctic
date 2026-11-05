"""
Plot every matched SLSTR / AWS pair in collocations_slstr_aws.csv.
Same figure as plot_collocation.py, one PNG per CSV row.

Top row:    SLSTR ice water path (full granule) | AWS frozen water path (full file)
Bottom row: the same, but only in the approximate shared grid cells
            (GRID_RES degree lat/lon cells containing both valid SLSTR pixels
            and AWS footprints observed within +/- MAX_DT of the granule)
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#import cmcrameri as cmcr
import cmocean as cmc
import matplotlib.ticker as mticker

plt.style.use("../plotstyling.mplstyle")

CSV = "collocations_slstr_aws.csv"
OUT_DIR = "../figures/cci_comparison"
AWS_VAR = "fwp_mean"    # name of the frozen water path variable in the AWS files
GRID_RES = 0.5          # grid cell size in degrees, same as in find_collocations.py
MAX_DT = np.timedelta64(60, "m")  # max AWS - SLSTR time difference, same as in find_collocations.py


def style_map(ax, scale="50m"):
    """Same look as the collocation maps, at a fixed level of detail."""
    ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="lightgrey", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale(scale), facecolor="dimgrey", zorder=0)
    ax.add_feature(cfeature.BORDERS.with_scale(scale), edgecolor="white", linewidth=0.5, zorder=1)
    ax.coastlines(resolution=scale, color="white", linewidth=0.8, zorder=10)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, x_inline=False, y_inline=False)
    # denser lines, so a small extent still gets several of each
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 10))
    gl.ylocator = mticker.FixedLocator(np.arange(50, 91, 5))
    # plain "30E" / "70N" labels (no degree symbol)
    gl.xformatter = LongitudeFormatter(degree_symbol="")
    gl.yformatter = LatitudeFormatter(degree_symbol="")
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}


def cell_keys(lat, lon, res=GRID_RES):
    """One number per point identifying its res-degree lat/lon grid cell (NaN if invalid)."""
    return np.floor(lat / res) * 10000 + np.floor(lon / res)


def padded_extent(lat, lon, pad=2):
    """Map extent around the valid points, with pad degrees of room."""
    return [np.nanmin(lon) - pad, np.nanmax(lon) + pad,
            max(np.nanmin(lat) - pad, 50), min(np.nanmax(lat) + pad, 90)]


fwp_norm = LogNorm(vmin=1e-2, vmax=1e0)
geo = ccrs.PlateCarree()

# values below vmin, including 0, are drawn in the lowest colour of the colormap
# (a log scale cannot show 0 itself, so zeros are nudged just below vmin)
ice_cmap = cmc.cm.ice.copy()
ice_cmap.set_under(ice_cmap(0))


def for_log(v):
    return np.where(v == 0, 1e-6, v)


def plot_pair(row, idx):
    """Make and save the 2x2 figure for one CSV row. Returns False if skipped."""

    # SLSTR ice water path, daylight only; g/m2 to kg/m2 to match AWS
    # (decode_times=False: the Julian Day time units do not decode correctly)
    #   ice (phase 2)                         -> cwp
    #   clear / liquid / aerosol (0, 1, 3-10) -> 0
    #   no phase (NaN), not daylight          -> NaN
    #   ice with NaN cwp (failed retrieval)   -> NaN
    with xr.open_dataset(row["cloud_file"], decode_times=False) as ds:
        phase = ds["phase"]
        no_ice = phase.notnull() & (phase != 2)
        iwp = xr.where(phase == 2, ds["cwp"] / 1000.0, xr.where(no_ice, 0.0, np.nan))
        iwp = iwp.where(ds["illum"] == 1)
        cci_lat, cci_lon, cci_val = ds["lat"].values, ds["lon"].values, iwp.values

    with xr.open_dataset(row["aws_file"]) as ds_aws:
        fwp = ds_aws[AWS_VAR]
        if "quantile" in fwp.dims:
            fwp = fwp.sel(quantile=0.5, method="nearest")  # median
        aws_lat = ds_aws["latitude"].values
        aws_lon = ds_aws["longitude"].values
        aws_val = fwp.values
        aws_time = ds_aws["scan"].broadcast_like(ds_aws["latitude"]).values

    # ---- AWS footprints observed within +/- MAX_DT of the SLSTR granule ----
    t0 = np.datetime64(pd.Timestamp(row["cloud_start"]))
    t1 = np.datetime64(pd.Timestamp(row["cloud_end"]))
    aws_in_time = (aws_time >= t0 - MAX_DT) & (aws_time <= t1 + MAX_DT)

    # ---- shared grid cells: valid SLSTR pixels (ice or 0) AND close-in-time AWS footprints ----
    cci_keys = cell_keys(cci_lat, cci_lon)
    aws_keys = cell_keys(aws_lat, aws_lon)
    valid = np.isfinite(cci_val)
    shared = np.intersect1d(cci_keys[valid], aws_keys[np.isfinite(aws_keys) & aws_in_time])
    if shared.size == 0:
        print("  no shared grid cells, skipped")
        return False

    cci_shared = np.where(np.isin(cci_keys, shared), cci_val, np.nan)
    aws_sel = np.isin(aws_keys, shared) & aws_in_time
    aws_shared = np.where(aws_sel, aws_val, np.nan)

    # ---- map extents ----
    # top row: the SLSTR granule; bottom row: the shared area
    extent_full = padded_extent(cci_lat, cci_lon)
    in_shared = np.isfinite(cci_shared)
    extent_shared = padded_extent(cci_lat[in_shared], cci_lon[in_shared], pad=1)

    fig, ((ax_cci, ax_aws), (ax_cci_s, ax_aws_s)) = plt.subplots(
        2, 2, figsize=(26, 24), subplot_kw={"projection": ccrs.NorthPolarStereo()}
    )

    time_cci = f"{row['cloud_start'][:16]} to {row['cloud_end'][11:16]} UTC"
    time_aws = f"{row['aws_start'][:16]} to {row['aws_end'][:16]} UTC"

    # ---- top row: full data ----
    ax = ax_cci
    style_map(ax)
    ax.set_extent(extent_full, crs=geo)
    mesh = ax.pcolormesh(cci_lon, cci_lat, for_log(cci_val), norm=fwp_norm, cmap=ice_cmap,
                         transform=geo, zorder=2)
    ax.set_title(f"SLSTR ice water path, daylight\n{time_cci}")

    ax = ax_aws
    style_map(ax)
    # scatter instead of pcolormesh: the AWS lat/lon contain NaNs, which
    # pcolormesh cannot handle (scatter just skips those points)
    ax.scatter(aws_lon.ravel(), aws_lat.ravel(), c=for_log(aws_val.ravel()), s=20, ec="none",
               norm=fwp_norm, cmap=ice_cmap, transform=geo, zorder=2)
    ax.set_title(f"AWS frozen water path\n{time_aws}")

    # ---- bottom row: shared grid cells only ----
    ax = ax_cci_s
    style_map(ax)
    ax.set_extent(extent_shared, crs=geo)
    ax.pcolormesh(cci_lon, cci_lat, for_log(cci_shared), norm=fwp_norm, cmap=ice_cmap,
                  transform=geo, zorder=2)
    ax.set_title(f"SLSTR, shared cells only\n{time_cci}")

    ax = ax_aws_s
    style_map(ax)
    ax.set_extent(extent_shared, crs=geo)
    ok = np.isfinite(aws_shared)
    ax.scatter(aws_lon[ok], aws_lat[ok], c=for_log(aws_shared[ok]), s=60, ec="none",
               norm=fwp_norm, cmap=ice_cmap, transform=geo, zorder=2)
    # time span of the AWS footprints shown in this panel
    span_lo = np.datetime_as_string(aws_time[aws_sel].min(), unit="m").replace("T", " ")
    span_hi = np.datetime_as_string(aws_time[aws_sel].max(), unit="m").replace("T", " ")
    ax.set_title(f"AWS, shared cells only\n{span_lo} to {span_hi[11:]} UTC")

    fig.colorbar(mesh, ax=[ax_cci, ax_aws, ax_cci_s, ax_aws_s], orientation="horizontal",
                 shrink=0.5, pad=0.04, extend="both", label=r"FWP [kg m$^{-2}$]")

    fig.savefig(f"{OUT_DIR}/single_case_{idx}.png", dpi=300,
                bbox_inches="tight", pad_inches=0.1)
    # close each figure, otherwise memory fills up over many plots
    plt.close(fig)
    return True


# ---- loop over all collocations ----
df = pd.read_csv(CSV)
n_done = 0
for idx, row in df.iterrows():
    print(f"[{idx + 1}/{len(df)}] {row['cloud_start'][:16]}  "
          f"{row['cloud_file'].split('/')[-1]}")
    try:
        if plot_pair(row, idx):
            n_done += 1
    except Exception as e:
        # one bad file should not stop the whole loop
        print(f"  failed: {e}")
        plt.close("all")

print(f"{n_done} of {len(df)} figures saved to {OUT_DIR}")