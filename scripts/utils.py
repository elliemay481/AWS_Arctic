import warnings

import numpy as np
import xarray as xr
import pickle
import pandas as pd


KM_PER_DEG = 111.2
EARTH_RADIUS_KM = KM_PER_DEG * 180.0 / np.pi   # consistent with KM_PER_DEG

AWS_ZM_VAR = "fwp_zm_mean"   # AWS mean mass height of the frozen water [m]

AWS_QUANTILE_KEYS = ("fwp_q1", "fwp_q16", "fwp_q84", "fwp_q99",
                     "lwp_q1", "lwp_q16", "lwp_q84", "lwp_q99")


def calendar_months(date_str):
    """Calendar months (1-12) in a name like '2025-12_to_2026-02' or '2025-07_2025-08_2026-06'."""
    tokens = date_str.split("_")
    periods = []
    i = 0
    while i < len(tokens):
        if i + 2 < len(tokens) and tokens[i + 1] == "to":      # a range
            periods.extend(pd.period_range(tokens[i], tokens[i + 2], freq="M"))
            i += 3
        else:                                                  # a single month
            periods.append(pd.Period(tokens[i], freq="M"))
            i += 1
    return sorted({p.month for p in periods})


def load_file_pairs(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_aws(aws_filepath, start=0, end=400):
    with xr.open_dataset(aws_filepath) as ds:
        lon_da = ds.longitude[start:end, :]
        scan_2d = xr.broadcast(lon_da["scan"], lon_da)[0]
        n_scan, n_fov = lon_da.shape
        aws = {
            "lon": lon_da.values.ravel(),
            "lat": ds.latitude[start:end, :].values.ravel(),
            "scan": scan_2d.values.ravel(),
            # across-track position (column) of each pixel, 0 to n_fov - 1
            "fov": np.broadcast_to(np.arange(n_fov)[None, :], (n_scan, n_fov)).ravel(),
        }
        for var in ("fwp", "lwp"):
            aws[var] = ds[f"{var}_mean"][start:end, :].values.ravel()
            quantiles = ds[f"{var}_quantiles"][start:end, :]
            for q in (1, 16, 84, 99):
                aws[f"{var}_q{q}"] = quantiles.sel(quantile=q / 100, method="nearest").values.ravel()

        # mean mass height; missing from the file -> all NaN, so nothing breaks
        if AWS_ZM_VAR in ds:
            aws["zm"] = ds[AWS_ZM_VAR][start:end, :].values.ravel()
        else:
            print(f"    warning: {AWS_ZM_VAR} not in {aws_filepath}, AWS Zm set to NaN")
            aws["zm"] = np.full(aws["lon"].shape, np.nan)
    return aws


def mass_weighted_height(iwc, iwp, height):
    """Mean mass height of each profile, in the units of height.

        zm = sum(z * iwc * dz) / iwp

    Missing water content counts as none. Profiles with no water give NaN.
    """
    z = np.broadcast_to(height, iwc.shape)
    dz = np.abs(np.gradient(z, axis=-1))
    iwc = np.where(np.isfinite(iwc), iwc, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        zm = np.nansum(z * iwc * dz, axis=-1) / iwp
    zm[~(iwp > 0)] = np.nan
    return zm


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

    # mean mass height of the ice [m], the EarthCARE counterpart of AWS zm
    ea["zm"] = mass_weighted_height(ea["iwc"], ea["iwp"], ea["height"])
    return ea


# ============================================================
# COLLOCATION
# ============================================================
def distance_km(lat1, lon1, lat2, lon2):
    """Distance along the Earth's surface [km] between points given in degrees.

    Haversine formula: exact on a sphere, works across the dateline and near
    the pole, and broadcasts like any numpy operation.

    """
    lat1, lon1, lat2, lon2 = (np.deg2rad(v) for v in (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _mean_profile(profiles):
    """Mean of several profiles (profile, level), missing values counting as none.

    Levels missing in every profile stay missing.
    """
    valid = np.isfinite(profiles)
    mean = np.where(valid, profiles, 0.0).mean(axis=0)
    mean[~valid.any(axis=0)] = np.nan
    return mean


def _nanmean_quiet(values, axis=None):
    """np.nanmean without the warning for all-NaN input (it gives NaN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


def colocate_nearest_profile(aws, ea, max_dist_km=15.0, avg_dist_km=15.0):
    """
    Collocate AWS pixels with EarthCARE, one AWS pixel per fov.

    For each fov (across-track position) in turn:

    1. Selecting the AWS pixel (max_dist_km)
       The distance [km] from every pixel in this fov to every EarthCARE profile
       is calculated. The pixel closest to the EarthCARE track is kept, if it is
       within max_dist_km of a profile. Over all fovs, this gives a strip of
       AWS pixels along the track, at most one per fov.

    2. The EarthCARE values for that pixel (avg_dist_km)
       All EarthCARE profiles within avg_dist_km of the pixel are averaged:
       IWP and LWP are the mean over those profiles, the IWC and LWC profiles
       are the mean profiles (missing values counting as none), and Zm is
       the mean mass height of the mean IWC profile. Neighbouring pixels can
       share profiles. A pixel with no profile within avg_dist_km is dropped,
       so if avg_dist_km < max_dist_km, it also limits the match distance.
       avg_dist_km=None uses only the nearest profile instead of an average.

    Only AWS pixels with a valid position and FWP are considered. aws must
    hold the "fov" of each pixel, as returned by load_aws.

    Returns one entry per selected AWS pixel, sorted by fov, with the same keys
    as before, plus "fov" and "n_ea_profiles" (number of profiles averaged).
    dist_km is the distance from the pixel to its nearest EarthCARE profile.
    """
    out = {key: [] for key in (
        "lat", "lon", "scan", "fov", "aws_fwp", "ea_iwp",
        "ea_iwc", "aws_lwp", "ea_lwp", "ea_lwc",
        *(f"aws_{key}" for key in AWS_QUANTILE_KEYS),
        "aws_zm", "ea_zm",
        "ea_height", "dist_km", "n_ea_profiles",
        "median_distance_to_ea", "mean_distance_to_ea"
    )}

    if "fov" not in aws:
        raise KeyError("aws has no 'fov' entry; load it with load_aws")

    # just to make sure all valid
    ea_ok = np.flatnonzero(np.isfinite(ea["lat"]) & np.isfinite(ea["lon"]))
    aws_ok = np.isfinite(aws["lat"]) & np.isfinite(aws["lon"]) & np.isfinite(aws["fwp"])
    ea_lat = ea["lat"][ea_ok]
    ea_lon = ea["lon"][ea_ok]

    if ea_ok.size > 0:
        # we iterate over aws fovs, since the earthcare tracks tend to cross many fovs and fewer scan rows
        for fov in np.unique(aws["fov"][aws_ok]):
            
            aws_points = np.flatnonzero(aws_ok & (aws["fov"] == fov))

            # calculate distance [km] between the aws observations at the given fov to every earthcare observation
            d = distance_km(aws["lat"][aws_points][:, None], aws["lon"][aws_points][:, None],
                            ea_lat[None, :], ea_lon[None, :])
            # this is a 2D array of distances: rows for different aws scans and columns for different earthcare observations

            nearest = d.min(axis=1)          # find minimum of the distance array
            k = np.argmin(nearest)           # this gives us the row with the smallest distance, i.e. an aws scan
            if nearest[k] > max_dist_km:     # earthcare track does not pass close to this fov
                continue
            j = aws_points[k]                    # we select this along-track row for the given fov

            # now we've selected which aws observation to use, we average over the earthcare data

            distance_earthcare_from_aws_sample = d[k]    # as the row d[k] contains distance to all earthcare points
            # select earthcare cases with distance is less than specified amount
            within = distance_earthcare_from_aws_sample <= avg_dist_km
            ea_samples = ea_ok[within]                                   # indices in the full EarthCARE arrays
            sample_dist = distance_earthcare_from_aws_sample[within]     # their distances [km]
            
            if ea_samples.size == 0:
                continue

            # save closest earthcare profile
            closest_ea_idx = ea_samples[np.argmin(sample_dist)]
            out["ea_iwc"].append(ea["iwc"][closest_ea_idx])
            out["ea_lwc"].append(ea["lwc"][closest_ea_idx])
            out["ea_height"].append(ea["height"][closest_ea_idx])

            # other earthcare data, averaged over nearest samples
            iwp = ea["iwp"][ea_samples]
            iwp = iwp[np.isfinite(iwp)]
            out["ea_iwp"].append(iwp.mean() if iwp.size else np.nan)   # zeros included
            out["ea_lwp"].append(_nanmean_quiet(ea["lwp"][ea_samples]))

            zm = mass_weighted_height(ea["iwc"][ea_samples], ea["iwp"][ea_samples], ea["height"][ea_samples])
            
            out["ea_zm"].append(_nanmean_quiet(zm))

            # as a form of quality flag
            out["n_ea_profiles"].append(ea_samples.size)
            median_distance = np.median(sample_dist)
            mean_distance = np.mean(sample_dist)
            out["median_distance_to_ea"].append(median_distance)
            out["mean_distance_to_ea"].append(mean_distance)

            # the AWS obs
            out["lat"].append(aws["lat"][j])
            out["lon"].append(aws["lon"][j])
            out["scan"].append(aws["scan"][j])
            out["fov"].append(aws["fov"][j])
            out["aws_fwp"].append(aws["fwp"][j])
            out["aws_lwp"].append(aws["lwp"][j])
            out["aws_zm"].append(aws["zm"][j])
            for key in AWS_QUANTILE_KEYS:
                out[f"aws_{key}"].append(aws[key][j])
            out["dist_km"].append(nearest[k])

    return {key: np.asarray(val) for key, val in out.items()}


def colocate_pair_nearest_profile(aws_filepath, ea_filepath, start=0, end=400,
                                  max_dist_km=15.0, avg_dist_km=15.0):
    aws = load_aws(aws_filepath, start=start, end=end)
    ea = load_earthcare(ea_filepath)
    return colocate_nearest_profile(aws, ea, max_dist_km=max_dist_km,
                                    avg_dist_km=avg_dist_km)