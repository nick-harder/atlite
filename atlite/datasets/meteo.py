# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2021 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Module for downloading and curating data from ECMWFs ERA5 dataset (via CDS).

For further reference see
https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
"""

import logging
import os
import warnings
import weakref
from tempfile import mkstemp

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
from dask import compute, delayed
from dask.array import arctan2, sqrt
from numpy import atleast_1d
import time
import openmeteo_requests
import requests_cache
from retry_requests import retry

from ..gis import maybe_swap_spatial_dims
from ..pv.solar_position import SolarPosition

# Null context for running a with statements wihout any context
try:
    from contextlib import nullcontext
except ImportError:
    # for Python verions < 3.7:
    import contextlib

    @contextlib.contextmanager
    def nullcontext():
        yield


logger = logging.getLogger(__name__)

# Global variables for rate limiting
API_LIMIT = 6 # 600 requests per minute allowed but each API call now makes 100 requests
TIME_WINDOW = 60  # seconds
request_count = 0
start_time = time.time()

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

base_url = "https://api.open-meteo.com/v1/forecast"

# Model and CRS Settings
crs = 4326

features = {
    "all": [
        "height",
        "wnd80m",
        "wnd_azimuth",
        "roughness",
        "influx_toa",
        "influx_direct",
        "influx_diffuse",
        "albedo",
        "solar_altitude",
        "solar_azimuth",
        "temperature",
        "soil temperature",
    ],
    "height": ["height"],
    "wind": ["wnd80m", "wnd_azimuth", "roughness"],
    "influx": [
        "influx_toa",
        "influx_direct",
        "influx_diffuse",
        "albedo",
        "solar_altitude",
        "solar_azimuth",
    ],
    "temperature": ["temperature", "soil temperature"],
}

static_features = {"height"}


def _add_height(ds):
    """
    Convert geopotential 'z' to geopotential height following [1].

    References
    ----------
    [1] ERA5: surface elevation and orography, retrieved: 10.02.2019
    https://confluence.ecmwf.int/display/CKB/ERA5%3A+surface+elevation+and+orography

    """
    g0 = 9.80665
    z = ds["z"]
    if "time" in z.coords:
        z = z.isel(time=0, drop=True)
    ds["height"] = z / g0
    ds = ds.drop_vars("z")
    return ds


def _rename_and_clean_coords(ds, add_lon_lat=True):
    """
    Rename 'longitude' and 'latitude' columns to 'x' and 'y' and fix roundings.

    Optionally (add_lon_lat, default:True) preserves latitude and
    longitude columns as 'lat' and 'lon'.
    """
    ds = ds.rename({"longitude": "x", "latitude": "y"})
    if "valid_time" in ds.sizes:
        ds = ds.rename({"valid_time": "time"}).unify_chunks()
    # round coords since cds coords are float32 which would lead to mismatches
    ds = ds.assign_coords(
        x=np.round(ds.x.astype(float), 5), y=np.round(ds.y.astype(float), 5)
    )
    ds = maybe_swap_spatial_dims(ds)
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    ds = ds.drop_vars(["expver", "number"], errors="ignore")

    return ds

def get_data_all(retrieval_params):
    """Get all data from meteo API for given retrieval parameters at once to save requests and runtime."""
    times = retrieval_times(retrieval_params["coords"], static=True)
    del retrieval_params["coords"]

    ds = retrieve_meteo_data(
        variable=[
            "wind_speed_80m",
            "wind_direction_80m",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
            "terrestrial_radiation",
            "temperature_2m",
            "soil_temperature_54cm",
        ],
        **retrieval_params,
    )

    ds_era5 = retrieve_era5_data(
        variable=[
            "forecast_surface_roughness",
            "toa_incident_solar_radiation",
            "forecast_albedo",
        ],
        **retrieval_params,
    )

    ds_era5_fal = (
        ds_era5["fal"]
        .interp(
            time=ds.time.values,
            method="nearest",
            kwargs={"fill_value": "extrapolate"},
        )
        .chunk(chunks=retrieval_params["chunks"])
    )

    ds_era5_fsr = (
        ds_era5["fsr"]
        .interp(
            time=ds.time.values,
            method="nearest",
            kwargs={"fill_value": "extrapolate"},
        )
        .chunk(chunks=retrieval_params["chunks"])
    )

    ds_era5_tisr = (
        ds_era5["tisr"]
        .interp(
            time=ds.time.values,
            method="nearest",
            kwargs={"fill_value": "extrapolate"},
        )
        .chunk(chunks=retrieval_params["chunks"])
    )

    attrs = ds_era5.attrs
    ds_era5 = xr.merge([ds_era5_fal, ds_era5_fsr, ds_era5_tisr])
    ds_era5.attrs = attrs

    (
        retrieval_params["year"],
        retrieval_params["month"],
        retrieval_params["day"],
        retrieval_params["time"],
    ) = (times["year"], times["month"], times["day"], times["time"])

    z_era5 = retrieve_era5_data(variable="geopotential", **retrieval_params).chunk(
        chunks=retrieval_params["chunks"]
    )
    z_era5 = _add_height(z_era5)

    ds = xr.merge([ds, ds_era5, z_era5])

    ds = _rename_and_clean_coords(ds)

    ds = ds.rename(
        {
            "temperature_2m": "temperature",
            "soil_temperature_54cm": "soil temperature",
            "direct_radiation": "influx_direct",
            "diffuse_radiation": "influx_diffuse",
            "wind_speed_80m": "wnd80m",
            "wind_direction_80m": "wnd_azimuth",
            "fal": "albedo",
            "fsr": "roughness",
            "tisr": "influx_toa",
        }
    )

    ds = ds.drop_vars(
        ["shortwave_radiation", "direct_normal_irradiance", "terrestrial_radiation"]
    )

    # Convert from Celsius to Kelvin C -> K, by adding 273.15
    ds[["temperature", "soil temperature"]] = (
        ds[["temperature", "soil temperature"]] + 273.15
    )

    # Convert from energy to power J m**-2 -> W m**-2 and clip negative fluxes
    ds["influx_toa"] = ds["influx_toa"] / (60.0 * 60.0)

    ds["temperature"].attrs.update(units="K", long_name="2 metre temperature")
    ds["soil temperature"].attrs.update(units="K", long_name="Soil temperature 54cm")

    ds["wnd80m"].attrs.update(
        units="m s**-1", long_name="Wind speed at 80m above ground"
    )
    ds["wnd_azimuth"].attrs.update(
        units="degree", long_name="Wind direction at 80m above ground"
    )

    ds["influx_direct"].attrs.update(
        units="W m**-2", long_name="Surface direct solar radiation downwards"
    )
    ds["influx_diffuse"].attrs.update(
        units="W m**-2", long_name="Surface diffuse solar radiation downwards"
    )
    ds["influx_toa"].attrs.update(
        units="W m**-2", long_name="TOA incident solar radiation"
    )

    # unify_chunks() is necessary to avoid a bug in xarray
    ds = ds.unify_chunks()

    # ERA5 variables are mean values for previous hour, i.e. 13:01 to 14:00 are labelled as "14:00"
    # account by calculating the SolarPosition for the center of the interval for aggregation happens
    # see https://github.com/PyPSA/atlite/issues/158
    # Do not show DeprecationWarning from new SolarPosition calculation (#199)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        time_shift = pd.to_timedelta("-30 minutes")
        sp = SolarPosition(ds, time_shift=time_shift)
    sp = sp.rename({v: f"solar_{v}" for v in sp.data_vars})

    ds = xr.merge([ds, sp])

    return ds


def sanitize_all(ds):
    """Sanitize all retrieved data."""
    ds["roughness"] = ds["roughness"].where(ds["roughness"] >= 0.0, 2e-4)

    for a in ("influx_direct", "influx_diffuse", "influx_toa"):
        ds[a] = ds[a].clip(min=0.0)

    return ds


def get_data_wind(retrieval_params):
    """Get wind data for given retrieval parameters."""
    ds = retrieve_meteo_data(
        variable=[
            "wind_speed_80m",
            "wind_direction_80m",
        ],
        **retrieval_params,
    )

    fsr = retrieve_era5_data(
        variable=["forecast_surface_roughness"],
        **retrieval_params,
    )

    fsr = fsr.interp(
        time=ds.time.values,
        method="nearest",
        kwargs={"fill_value": "extrapolate"},
    )

    ds = xr.merge([ds, fsr])

    ds = _rename_and_clean_coords(ds)

    ds = ds.rename(
        {
            "wind_speed_80m": "wnd80m",
            "wind_direction_80m": "wnd_azimuth",
            "fsr": "roughness",
        }
    )

    ds.wnd80m.attrs.update(units="m s**-1", long_name="Wind speed at 80m above ground")
    ds.wnd_azimuth.attrs.update(
        units="degree", long_name="Wind direction at 80m above ground"
    )

    return ds


def sanitize_wind(ds):
    """Sanitize retrieved wind data."""
    ds["roughness"] = ds["roughness"].where(ds["roughness"] >= 0.0, 2e-4)
    return ds


def get_data_influx(retrieval_params):
    """Get influx data for given retrieval parameters."""
    ds = retrieve_meteo_data(
        variable=[
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
            "terrestrial_radiation",
        ],
        **retrieval_params,
    )

    fal = retrieve_era5_data(
        variable=["forecast_albedo"],
        **retrieval_params,
    )

    tisr = retrieve_era5_data(
        variable=["toa_incident_solar_radiation"],
        **retrieval_params,
    )

    fal = fal.interp(
        time=ds.time.values,
        method="nearest",
        kwargs={"fill_value": "extrapolate"},
    )

    tisr = tisr.interp(
        time=ds.time.values,
        method="nearest",
        kwargs={"fill_value": "extrapolate"},
    )

    ds = xr.merge([ds, fal, tisr])

    ds = _rename_and_clean_coords(ds)

    ds = ds.rename(
        {
            "direct_radiation": "influx_direct",
            "diffuse_radiation": "influx_diffuse",
            "tisr": "influx_toa",
            "fal": "albedo",
        }
    )

    ds = ds.drop_vars(
        ["shortwave_radiation", "terrestrial_radiation", "direct_normal_irradiance"]
    )

    # Convert from energy to power J m**-2 -> W m**-2 and clip negative fluxes
    ds["influx_toa"] = ds["influx_toa"] / (60.0 * 60.0)

    ds.influx_direct.attrs.update(
        units="W m**-2", long_name="Surface direct solar radiation downwards"
    )
    ds.influx_diffuse.attrs.update(
        units="W m**-2", long_name="Surface diffuse solar radiation downwards"
    )
    ds.influx_toa.attrs.update(
        units="W m**-2", long_name="TOA incident solar radiation"
    )

    # ERA5 variables are mean values for previous hour, i.e. 13:01 to 14:00 are labelled as "14:00"
    # account by calculating the SolarPosition for the center of the interval for aggregation happens
    # see https://github.com/PyPSA/atlite/issues/158
    # Do not show DeprecationWarning from new SolarPosition calculation (#199)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        time_shift = pd.to_timedelta("-30 minutes")
        sp = SolarPosition(ds, time_shift=time_shift)
    sp = sp.rename({v: f"solar_{v}" for v in sp.data_vars})

    ds = xr.merge([ds, sp])

    return ds


def sanitize_influx(ds):
    """Sanitize retrieved influx data."""
    for a in ("influx_direct", "influx_diffuse", "influx_toa"):
        ds[a] = ds[a].clip(min=0.0)
    return ds


def get_data_temperature(retrieval_params):
    """Get wind temperature for given retrieval parameters."""
    ds = retrieve_meteo_data(
        variable=["temperature_2m", "soil_temperature_54cm"], **retrieval_params
    )

    ds = _rename_and_clean_coords(ds)
    ds = ds.rename(
        {"temperature_2m": "temperature", "soil_temperature_54cm": "soil temperature"}
    )

    # Convert from Celsius to Kelvin C -> K, by adding 273.15
    ds = ds + 273.15

    ds["temperature"].attrs.update(units="K", long_name="2 metre temperature")
    ds["soil temperature"].attrs.update(units="K", long_name="Soil temperature 54cm")

    return ds


def get_data_height(retrieval_params):
    """Get height data for given retrieval parameters."""
    ds = retrieve_era5_data(variable="geopotential", **retrieval_params)

    ds = _rename_and_clean_coords(ds)
    ds = _add_height(ds)

    return ds


def _area(coords):
    # North, West, South, East. Default: global
    x0, x1 = coords["x"].min().item(), coords["x"].max().item()
    y0, y1 = coords["y"].min().item(), coords["y"].max().item()
    return [y1, x0, y0, x1]


def noisy_unlink(path):
    """
    Delete file at given path.
    """
    logger.debug(f"Deleting file {path}")
    try:
        os.unlink(path)
    except PermissionError:
        logger.error(f"Unable to delete file {path}, as it is still in use.")


def retrieve_meteo_data(product, chunks=None, tmpdir=None, lock=None, **updates):
    """
    Download meteo data using Open Meteo Python API Client 
    """
    request = {"product_type": "meteo_api", "format": "direct_download"}
    request.update(updates)

    # Generate latitude and longitude grid
    g_lat = np.arange(
        request["area"][2], request["area"][0] + request["grid"][0], request["grid"][0]
    )
    g_lon = np.arange(
        request["area"][1], request["area"][3] + request["grid"][1], request["grid"][1]
    )

    era5_coords_lats = [lat for lat in g_lat for lon in g_lon]
    era5_coords_lons = [lon for lat in g_lat for lon in g_lon]

    # Precompute values that don't change in the loop
    start_date = request['start'].strftime('%Y-%m-%d')
    end_date = request['end'].strftime('%Y-%m-%d')
    
    all_coords_df = None 

    # Define chunk size
    chunk_size = 50

    # Loop through the coordinates in chunks
    for i in range(0, len(era5_coords_lats), chunk_size):
        # Extract latitude and longitude slices for the current chunk
        lat_chunk = era5_coords_lats[i:i + chunk_size]
        lon_chunk = era5_coords_lons[i:i + chunk_size]

        # Define parameters for the API request
        params = {
            "latitude": lat_chunk,
            "longitude": lon_chunk,
            "hourly": request["variable"],
            "wind_speed_unit": "ms",
            "start_date": start_date,
            "end_date": end_date,
        }

        # Apply rate limiting
        apply_rate_limiting()

        #Make request and retrieve responses from API
        responses = openmeteo.weather_api(base_url, params=params)

        #Parse responses into DataFrame
        coords_df = parse_meteo_responses(responses, params)
        
        # Concatenate data
        if all_coords_df is None:
            all_coords_df = coords_df
        else:
            all_coords_df = pd.concat([all_coords_df, coords_df])

    # Convert to xarray
    ds = all_coords_df.to_xarray().chunk(chunks=chunks)

    return ds

def apply_rate_limiting():
    """Check and apply rate limiting based on the API limit."""
    global request_count, start_time

    # If we've reached the API limit, sleep for the remainder of the time window
    if request_count >= API_LIMIT:
        elapsed_time = time.time() - start_time
        if elapsed_time < TIME_WINDOW:
            time.sleep(TIME_WINDOW - elapsed_time)
        # Reset count and timer after sleeping
        request_count = 0
        start_time = time.time()

    # Increment the request count
    request_count += 1


def parse_meteo_responses(responses, params):
    """Parse Open Meteo API responses and retrieve data"""
    coords_df = None

    # Iterate through responses (one response per coordinate)
    for x in range(len(responses)):
        response = responses[x]
        
        # Define timestamp indices
        range_start = pd.to_datetime(response.Hourly().Time(), unit = "s")
        range_end = pd.to_datetime(response.Hourly().TimeEnd(), unit = "s")
        date_range = pd.date_range(start=range_start, end=range_end, 
                                freq=pd.Timedelta(seconds = response.Hourly().Interval()),
                                inclusive="left")

        # Prepare DataFrame in which to store parameter values
        single_coord_df = pd.DataFrame(columns=params["hourly"])
        single_coord_df["time"] = date_range
        single_coord_df["latitude"] = params["latitude"][x]
        single_coord_df["longitude"] = params["longitude"][x]
        single_coord_df.set_index(["time", "latitude", "longitude"], inplace=True)
        
        # Iterate through meteo measurements and retrieve values
        for i, param in enumerate(params["hourly"]):
            single_coord_df[param] = response.Hourly().Variables(i).ValuesAsNumpy()
        
        # Concatenate data
        if coords_df is None:
            coords_df = single_coord_df
        else:
            coords_df = pd.concat([coords_df, single_coord_df])
    
    return coords_df


def retrieve_era5_data(product, chunks=None, tmpdir=None, lock=None, **updates):
    """
    Download data like ERA5 from the Climate Data Store (CDS).

    If you want to track the state of your request go to
    https://cds-beta.climate.copernicus.eu/requests?tab=all
    """
    request = {"product_type": "reanalysis", "format": "netcdf"}
    request.update(updates)

    assert {"year", "month", "variable"}.issubset(
        request
    ), "Need to specify at least 'variable', 'year' and 'month'"
    del request["start"], request["end"]

    client = cdsapi.Client(
        info_callback=logger.debug, debug=logging.DEBUG >= logging.root.level
    )
    result = client.retrieve("reanalysis-era5-single-levels", request)

    if lock is None:
        lock = nullcontext()

    with lock:
        fd, target = mkstemp(suffix=".nc", dir=tmpdir)
        os.close(fd)

        # Inform user about data being downloaded as "* variable (year-month)"
        timestr = f"{request['year']}-{request['month']}"
        variables = atleast_1d(request["variable"])
        varstr = "\n\t".join([f"{v} ({timestr})" for v in variables])
        logger.info(f"CDS: Downloading variables\n\t{varstr}\n")
        result.download(target)

    ds = xr.open_dataset(target, chunks=chunks or {})
    if tmpdir is None:
        logger.debug(f"Adding finalizer for {target}")
        weakref.finalize(ds._file_obj._manager, noisy_unlink, target)

    return ds


def retrieval_times(coords, static=False):
    """
    Get list of retrieval cdsapi arguments for time dimension in coordinates.

    If static is False, this function creates a query for each year in the
    time axis in coords. This ensures not running into query limits of the
    cdsapi. If static is True, the function return only one set of parameters
    for the very first time point.

    Parameters
    ----------
    coords : atlite.Cutout.coords

    Returns
    -------
    list of dicts witht retrieval arguments

    """
    time = coords["time"].to_index()
    if static:
        return {
            "year": str(time[0].year),
            "month": str(time[0].month),
            "day": str(time[0].day),
            "time": time[0].strftime("%H:00"),
        }

    times = []
    for year in time.year.unique():
        t = time[time.year == year]
        query = {
            "year": str(year),
            "month": list(t.month.unique()),
            "day": list(t.day.unique()),
            "time": ["%02d:00" % h for h in t.hour.unique()],
        }
        times.append(query)
    return times


def get_data(cutout, feature, tmpdir, lock=None, **creation_parameters):
    """
    Retrieve data from ECMWFs ERA5 dataset (via CDS).

    This front-end function downloads data for a specific feature and formats
    it to match the given Cutout.

    Parameters
    ----------
    cutout : atlite.Cutout
    feature : str
        Name of the feature data to retrieve. Must be in
        `atlite.datasets.era5.features`
    tmpdir : str/Path
        Directory where the temporary netcdf files are stored.
    **creation_parameters :
        Additional keyword arguments. The only effective argument is 'sanitize'
        (default True) which sets sanitization of the data on or off.

    Returns
    -------
    xarray.Dataset
        Dataset of dask arrays of the retrieved variables.

    """
    coords = cutout.coords

    sanitize = creation_parameters.get("sanitize", True)

    retrieval_params = {
        "product": "meteo_api_data",
        "area": _area(coords),
        "chunks": cutout.chunks,
        "grid": [cutout.dx, cutout.dy],
        "tmpdir": tmpdir,
        "lock": lock,
    }

    func = globals().get(f"get_data_{feature}")
    sanitize_func = globals().get(f"sanitize_{feature}")

    logger.info(f"Requesting data for feature {feature}...")

    def retrieve_once(time):
        ds = func({**retrieval_params, **time})
        if sanitize and sanitize_func is not None:
            ds = sanitize_func(ds)
        return ds

    time = coords["time"].to_index()

    start_date = time.min()
    end_date = time.max()

    if feature in static_features:
        datasets = retrieve_once(
            {
                **{"start": start_date, "end": end_date},
                **retrieval_times(coords, static=True),
            }
        )
    elif feature == "all":
        datasets = retrieve_once(
            {
                **{"start": start_date, "end": end_date, "coords": coords},
                **retrieval_times(coords)[0],
            }
        )
    else:
        datasets = retrieve_once(
            {**{"start": start_date, "end": end_date}, **retrieval_times(coords)[0]}
        )

    return datasets