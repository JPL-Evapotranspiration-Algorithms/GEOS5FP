"""
Reusable GEOS-5 FP point query helper.

Requires:
  conda install -c conda-forge xarray netcdf4 pandas numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import logging
import socket
import threading
import time
import warnings

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# Suppress xarray SerializationWarning about ambiguous reference dates in GEOS-5 FP files
warnings.filterwarnings('ignore', message='.*Ambiguous reference date string.*')

_SOCKET_TIMEOUT_LOCK = threading.Lock()


@dataclass
class PointQueryResult:
    data: xr.DataArray          # sliced variable at the point
    df: pd.DataFrame            # tidy dataframe (time-indexed)
    lat_used: float             # nearest grid lat
    lon_used: float             # nearest grid lon (as stored in dataset)
    url: str                    # actual OPeNDAP URL opened


def _to_naive_utc(ts: Union[str, pd.Timestamp, np.datetime64]) -> pd.Timestamp:
    """Convert input time to tz-naive UTC pandas Timestamp."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def _run_with_socket_timeout(timeout_seconds: float, func, *args, **kwargs):
    """Safely scope default socket timeout around netCDF4/OPeNDAP operations."""
    with _SOCKET_TIMEOUT_LOCK:
        original_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout_seconds)
        try:
            return func(*args, **kwargs)
        finally:
            socket.setdefaulttimeout(original_timeout)


def _retry_opendap_call(func, *, retries: int = 3, retry_delay: float = 10.0, **kwargs):
    """Execute an OPeNDAP query helper with retry/backoff logic."""
    last_error = None
    for attempt in range(retries):
        try:
            return func(**kwargs)
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            is_timeout = any(keyword in error_msg for keyword in [
                'timeout', '504', 'gateway', 'timed out', 'time-out'
            ])

            if attempt < retries - 1:
                if is_timeout:
                    logger.warning(f"OPeNDAP timeout/gateway error (attempt {attempt + 1}/{retries})")
                else:
                    logger.warning(f"OPeNDAP query failed (attempt {attempt + 1}/{retries}): {e}")

                delay = retry_delay * 2 if is_timeout else retry_delay
                logger.info(f"Retrying in {delay:.0f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"OPeNDAP query failed after {retries} attempts: {e}")
                raise

    raise last_error if last_error else RuntimeError("Query failed")


def query_geos5fp_point(
    dataset: str,
    variable: str,
    lat: float,
    lon: float,
    time_range: Optional[Tuple[Union[str, pd.Timestamp, np.datetime64],
                               Union[str, pd.Timestamp, np.datetime64]]] = None,
    *,
    base_url: str = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim",
    engine: str = "netcdf4",
    lon_convention: str = "auto",  # "auto", "neg180_180", "0_360"
    method: str = "nearest",
    dropna: bool = True,
    retries: int = 3,
    retry_delay: float = 10.0,
    timeout: float = 120.0,
) -> PointQueryResult:
    """
    Point-query a GEOS-5 FP OPeNDAP collection with retry logic.
    """
    url = f"{base_url}/{dataset}"
    return _retry_opendap_call(
        _query_geos5fp_point_impl,
        retries=retries,
        retry_delay=retry_delay,
        URL=url,
        dataset=dataset,
        variable=variable,
        lat=lat,
        lon=lon,
        time_range=time_range,
        engine=engine,
        lon_convention=lon_convention,
        method=method,
        dropna=dropna,
        timeout=timeout,
    )


def _query_geos5fp_point_impl(
    URL: str,
    dataset: str,
    variable: str,
    lat: float,
    lon: float,
    time_range: Optional[Tuple],
    engine: str,
    lon_convention: str,
    method: str,
    dropna: bool,
    timeout: float = 120.0,
) -> PointQueryResult:
    """Internal implementation of point query without retry logic."""
    def _execute():
        ds = xr.open_dataset(
            URL,
            engine=engine,
            cache=False,
            lock=False,
            decode_cf=False
        )

        # Handle longitude convention
        ds_lon = ds["lon"].values
        ds_max = float(np.nanmax(ds_lon))

        lon_used = lon
        if lon_convention == "0_360" or (lon_convention == "auto" and ds_max > 180):
            lon_used = lon % 360
        elif lon_convention == "neg180_180" or (lon_convention == "auto" and ds_max <= 180):
            lon_used = lon

        # Nearest gridpoint
        pt = ds.sel(lat=lat, lon=lon_used, method=method)

        if variable not in pt:
            raise KeyError(
                f"Variable {variable!r} not found in {dataset!r}. "
                f"Available: {list(pt.data_vars)}"
            )

        da = pt[variable]

        # Time slice if requested
        if time_range is not None:
            start, end = time_range
            start_n = _to_naive_utc(start)
            end_n = _to_naive_utc(end)
            import xarray.coding.times as xr_times
            time_var = da["time"]
            if hasattr(time_var, 'attrs') and 'units' in time_var.attrs:
                decoded_times = xr_times.decode_cf_datetime(
                    time_var.values,
                    units=time_var.attrs.get('units'),
                    calendar=time_var.attrs.get('calendar', 'standard')
                )
                da = da.assign_coords(time=decoded_times)
            da = da.sel(time=slice(start_n, end_n))

        time_values = da["time"].values

        if time_values.dtype.kind in ['i', 'f']:
            time_var = da["time"]
            if hasattr(time_var, 'attrs') and 'units' in time_var.attrs:
                import xarray.coding.times as xr_times
                time_index = pd.to_datetime(xr_times.decode_cf_datetime(
                    time_values,
                    units=time_var.attrs.get('units'),
                    calendar=time_var.attrs.get('calendar', 'standard')
                ))
            else:
                time_index = pd.to_datetime(time_values)
        else:
            time_index = pd.to_datetime(time_values)

        values = da.values

        df = pd.DataFrame({variable: values}, index=time_index)
        df.index.name = "time"

        if dropna:
            df = df.dropna()

        return PointQueryResult(
            data=da,
            df=df,
            lat_used=float(pt["lat"].values),
            lon_used=float(pt["lon"].values),
            url=URL,
        )

    return _run_with_socket_timeout(timeout, _execute)


def query_geos5fp_point_multi(
    dataset: str,
    variables: list[str],
    lat: float,
    lon: float,
    time_range: Optional[Tuple[Union[str, pd.Timestamp, np.datetime64],
                               Union[str, pd.Timestamp, np.datetime64]]] = None,
    *,
    base_url: str = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim",
    engine: str = "netcdf4",
    lon_convention: str = "auto",
    method: str = "nearest",
    dropna: bool = True,
    retries: int = 3,
    retry_delay: float = 10.0,
    timeout: float = 120.0,
) -> PointQueryResult:
    """
    Point-query multiple variables from a GEOS-5 FP OPeNDAP collection in a single request
    with retry logic and scoped socket timeout restoration.
    """
    url = f"{base_url}/{dataset}"
    return _retry_opendap_call(
        _query_geos5fp_point_multi_impl,
        retries=retries,
        retry_delay=retry_delay,
        url=url,
        dataset=dataset,
        variables=variables,
        lat=lat,
        lon=lon,
        time_range=time_range,
        engine=engine,
        lon_convention=lon_convention,
        method=method,
        dropna=dropna,
        timeout=timeout,
    )


def _query_geos5fp_point_multi_impl(
    url: str,
    dataset: str,
    variables: list[str],
    lat: float,
    lon: float,
    time_range: Optional[Tuple],
    engine: str,
    lon_convention: str,
    method: str,
    dropna: bool,
    timeout: float = 120.0,
) -> PointQueryResult:
    """Internal implementation of multi-variable point query without retry logic."""
    def _execute():
        logger.info(f"Opening OPeNDAP dataset: {url}")

        try:
            ds = xr.open_dataset(url, engine=engine, cache=False, lock=False, decode_cf=False)
            logger.info(f"Successfully opened dataset {dataset}")
        except Exception as e:
            logger.error(f"Failed to open dataset {url}: {e}")
            raise

        ds_lon = ds["lon"].values
        ds_max = float(np.nanmax(ds_lon))

        lon_used = lon
        if lon_convention == "0_360" or (lon_convention == "auto" and ds_max > 180):
            lon_used = lon % 360
        elif lon_convention == "neg180_180" or (lon_convention == "auto" and ds_max <= 180):
            lon_used = lon

        pt = ds.sel(lat=lat, lon=lon_used, method=method)

        missing = [v for v in variables if v not in pt]
        if missing:
            raise KeyError(
                f"Variables {missing!r} not found in {dataset!r}. "
                f"Available: {list(pt.data_vars)}"
            )

        pt = pt[variables]

        if time_range is not None:
            start, end = time_range
            start_n = _to_naive_utc(start)
            end_n = _to_naive_utc(end)
            import xarray.coding.times as xr_times
            time_var = pt["time"]
            if hasattr(time_var, 'attrs') and 'units' in time_var.attrs:
                decoded_times = xr_times.decode_cf_datetime(
                    time_var.values,
                    units=time_var.attrs.get('units'),
                    calendar=time_var.attrs.get('calendar', 'standard')
                )
                pt = pt.assign_coords(time=decoded_times)
            pt = pt.sel(time=slice(start_n, end_n))

        df = pt.to_dataframe()

        if 'lat' in df.index.names and 'lon' in df.index.names:
            df = df.reset_index(level=["lat", "lon"], drop=True)

        if 'time' not in df.index.names and 'time' in df.columns:
            df = df.set_index('time')

        df.index.name = "time"

        if dropna:
            df = df.dropna()

        return PointQueryResult(
            data=pt,
            df=df,
            lat_used=float(pt["lat"].values),
            lon_used=float(pt["lon"].values),
            url=url,
        )

    return _run_with_socket_timeout(timeout, _execute)
