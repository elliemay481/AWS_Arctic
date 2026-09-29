import numpy as np
import xarray as xr
import pickle

KM_PER_DEG = 111.2


def load_file_pairs(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_aws(aws_filepath, start=0, end=400):
    with xr.open_dataset(aws_filepath) as ds:
        lon_da = ds.longitude[start:end, :]
        scan_2d = xr.broadcast(lon_da["scan"], lon_da)[0]
        aws = {
            "lon": lon_da.values.ravel(),
            "lat": ds.latitude[start:end, :].values.ravel(),
            "scan": scan_2d.values.ravel(),
        }
        for var in ("fwp", "lwp"):
            aws[var] = ds[f"{var}_mean"][start:end, :].values.ravel()
            quantiles = ds[f"{var}_quantiles"][start:end, :]
            for q in (16, 84):
                aws[f"{var}_q{q}"] = quantiles.sel(quantile=q / 100, method="nearest").values.ravel()
    return aws


def load_earthcare(ea_filepath):
    ea_data = xr.open_datatree(ea_filepath, engine="h5netcdf")
    sci = ea_data["ScienceData"]
    ea = {
        "lon": np.asarray(sci.longitude.values),
        "lat": np.asarray(sci.latitude.values),
        "iwp": np.asarray(sci["ice_water_path"].values),
        "height": np.asarray(sci["height"].values),
        "iwc": np.asarray(sci["ice_water_content"].values),
        "lwc": np.asarray(sci["liquid_water_content"].values),
    }
    ea_data.close()

    # liquid water path (kg m-2): integrate LWC (kg m-3) over height (m) in
    # each profile. Missing LWC counts as no liquid; abs() because the
    # heights may run from the top down, which would flip the sign.
    lwc = np.where(np.isfinite(ea["lwc"]), ea["lwc"], 0.0)
    ea["lwp"] = np.abs(np.trapezoid(lwc, x=ea["height"], axis=1))

    # profiles with no valid LWC at all stay missing rather than 0
    ea["lwp"][~np.isfinite(ea["lwc"]).any(axis=1)] = np.nan
    return ea


def colocate_nearest_profile(aws, ea, max_dist_km=5.0):
    """
    For each AWS pixel, take the nearest EarthCARE profile within max_dist_km.
    Returns one entry per AWS pixel with a match, including the full IWC and LWC profiles.
    """
    in_box = (
        (ea["lon"] >= np.nanmin(aws["lon"]))
        & (ea["lon"] <= np.nanmax(aws["lon"]))
        & (ea["lat"] >= np.nanmin(aws["lat"]))
        & (ea["lat"] <= np.nanmax(aws["lat"]))
    )
    ea_idx = np.flatnonzero(in_box)
    ea_lon = ea["lon"][ea_idx]
    ea_lat = ea["lat"][ea_idx]

    out = {"lat": [], "lon": [], "scan": [], "aws_fwp": [], "ea_iwp": [],
           "ea_iwc": [], "aws_lwp": [], "ea_lwp": [], "ea_lwc": [],
           "aws_fwp_q16": [], "aws_fwp_q84": [], "aws_lwp_q16": [], "aws_lwp_q84": [],
           "ea_height": [], "dist_km": []}

    if ea_idx.size > 0:
        lat_margin = max_dist_km / KM_PER_DEG
        candidates = np.flatnonzero(
            (aws["lat"] >= np.nanmin(ea_lat) - lat_margin)
            & (aws["lat"] <= np.nanmax(ea_lat) + lat_margin)
            & np.isfinite(aws["fwp"])
        )

        for j in candidates:
            dlat = ea_lat - aws["lat"][j]
            near = np.flatnonzero(np.abs(dlat) * KM_PER_DEG <= max_dist_km)
            if near.size == 0:
                continue

            dlon = ea_lon[near] - aws["lon"][j]
            dlon = (dlon + 180.0) % 360.0 - 180.0
            dx = dlon * np.cos(np.deg2rad(aws["lat"][j])) * KM_PER_DEG
            dy = dlat[near] * KM_PER_DEG
            d = np.sqrt(dx * dx + dy * dy)

            i = np.argmin(d)
            if d[i] > max_dist_km:
                continue
            p = ea_idx[near[i]]

            out["lat"].append(aws["lat"][j])
            out["lon"].append(aws["lon"][j])
            out["scan"].append(aws["scan"][j])
            out["aws_fwp"].append(aws["fwp"][j])
            out["ea_iwp"].append(ea["iwp"][p])
            out["ea_iwc"].append(ea["iwc"][p])
            out["aws_lwp"].append(aws["lwp"][j])
            out["ea_lwp"].append(ea["lwp"][p])
            out["ea_lwc"].append(ea["lwc"][p])
            out["ea_height"].append(ea["height"][p])
            out["dist_km"].append(d[i])
            for key in ("fwp_q16", "fwp_q84", "lwp_q16", "lwp_q84"):
                out[f"aws_{key}"].append(aws[key][j])

    return {key: np.asarray(val) for key, val in out.items()}


def colocate_pair_nearest_profile(aws_filepath, ea_filepath, start=0, end=400,
                                  max_dist_km=5.0):
    aws = load_aws(aws_filepath, start=start, end=end)
    ea = load_earthcare(ea_filepath)
    return colocate_nearest_profile(aws, ea, max_dist_km=max_dist_km)