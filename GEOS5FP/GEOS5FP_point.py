"""
Reusable GEOS-5 FP point query helper.

Requires:
  conda install -c conda-forge xarray netcdf4 pandas numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import xarray as xr

# Suppress xarray SerializationWarning about ambiguous reference dates in GEOS-5 FP files
warnings.filterwarnings('ignore', message='.*Ambiguous reference date string.*')


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
) -> PointQueryResult:
    """
    Point-query a GEOS-5 FP OPeNDAP collection.

    Parameters
    ----------
    dataset : str
        Collection name, e.g. "inst3_2d_asm_Nx".
    variable : str
        Variable name inside that collection, e.g. "t2m", "ps", "u10m".
    lat, lon : float
        Target coordinate in degrees.
    time_range : (start, end) or None
        Anything pandas.Timestamp can parse. If tz-aware, will be converted to UTC-naive.
        If None, returns full time series available in the collection.
    base_url : str
        Root OPeNDAP path up to /assim (or /tavg, etc).
    lon_convention : str
        - "auto": detect dataset lon range & convert input if needed
        - "neg180_180": assume dataset uses [-180, 180]
        - "0_360": assume dataset uses [0, 360)
    method : str
        Selection method for lat/lon (usually "nearest").

    Returns
    -------
    PointQueryResult
    """
    url = f"{base_url}/{dataset}"
    # Disable caching, locking, and CF decoding to speed up remote OPeNDAP access
    ds = xr.open_dataset(url, engine=engine, cache=False, lock=False, decode_cf=False)

    # Handle longitude convention
    ds_lon = ds["lon"].values
    ds_min, ds_max = float(np.nanmin(ds_lon)), float(np.nanmax(ds_lon))

    lon_used = lon
    if lon_convention == "0_360" or (lon_convention == "auto" and ds_max > 180):
        lon_used = lon % 360
    elif lon_convention == "neg180_180" or (lon_convention == "auto" and ds_max <= 180):
        # keep as-is for [-180,180] datasets
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
        # With decode_cf=False, time is numeric - need to decode for selection
        # Decode times for the selection operation
        import xarray.coding.times as xr_times
        time_var = da["time"]
        if hasattr(time_var, 'attrs') and 'units' in time_var.attrs:
            # Manually decode time for selection
            decoded_times = xr_times.decode_cf_datetime(
                time_var.values,
                units=time_var.attrs.get('units'),
                calendar=time_var.attrs.get('calendar', 'standard')
            )
            # Create a temporary DataArray with decoded times for selection
            da = da.assign_coords(time=decoded_times)
        da = da.sel(time=slice(start_n, end_n))

    # Get values directly instead of calling .load() to avoid hanging
    # This triggers data loading but seems more reliable with remote OPeNDAP
    values = da.values
    time_values = da["time"].values
    
    # Make tidy DataFrame - decode times if they're still numeric
    if time_values.dtype.kind in ['i', 'f']:  # numeric type
        # Need to manually decode
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
        url=url,
    )


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
) -> PointQueryResult:
    """
    Point-query multiple variables from a GEOS-5 FP OPeNDAP collection in a single request.
    
    This is significantly faster than calling query_geos5fp_point() multiple times because:
    1. Opens the dataset connection once
    2. Queries all variables in a single OPeNDAP request
    3. Reduces network overhead
    
    Parameters
    ----------
    dataset : str
        Collection name, e.g. "tavg1_2d_slv_Nx".
    variables : list[str]
        List of variable names inside that collection, e.g. ["t2m", "ps", "u10m"].
    lat, lon : float
        Target coordinate in degrees.
    time_range : (start, end) or None
        Anything pandas.Timestamp can parse. If tz-aware, will be converted to UTC-naive.
        If None, returns full time series available in the collection.
    base_url : str
        Root OPeNDAP path up to /assim (or /tavg, etc).
    lon_convention : str
        - "auto": detect dataset lon range & convert input if needed
        - "neg180_180": assume dataset uses [-180, 180]
        - "0_360": assume dataset uses [0, 360)
    method : str
        Selection method for lat/lon (usually "nearest").
    dropna : bool
        If True, drop rows where ANY variable is NaN.
        
    Returns
    -------
    PointQueryResult
        The `data` field contains an xarray.Dataset with all variables.
        The `df` field contains a pandas DataFrame with time index and one column per variable.
        
    Examples
    --------
    >>> result = query_geos5fp_point_multi(
    ...     dataset="tavg1_2d_slv_Nx",
    ...     variables=["t2m", "qv2m", "ps"],
    ...     lat=34.05,
    ...     lon=-118.25,
    ...     time_range=("2024-01-01", "2024-01-31")
    ... )
    >>> result.df.head()
                         t2m     qv2m          ps
    time                                        
    2024-01-01 00:30:00  285.2  0.0082  101325.0
    2024-01-01 01:30:00  284.8  0.0081  101330.0
    """
    url = f"{base_url}/{dataset}"
    # Disable caching, locking, and CF decoding to speed up remote OPeNDAP access
    ds = xr.open_dataset(url, engine=engine, cache=False, lock=False, decode_cf=False)
    
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
    
    # Validate all variables exist
    missing = [v for v in variables if v not in pt]
    if missing:
        raise KeyError(
            f"Variables {missing!r} not found in {dataset!r}. "
            f"Available: {list(pt.data_vars)}"
        )
    
    # Select only requested variables
    pt = pt[variables]
    
    # Time slice if requested
    if time_range is not None:
        start, end = time_range
        start_n = _to_naive_utc(start)
        end_n = _to_naive_utc(end)
        # With decode_cf=False, time is numeric - need to decode for selection
        # Decode times for the selection operation
        import xarray.coding.times as xr_times
        time_var = pt["time"]
        if hasattr(time_var, 'attrs') and 'units' in time_var.attrs:
            # Manually decode time for selection
            decoded_times = xr_times.decode_cf_datetime(
                time_var.values,
                units=time_var.attrs.get('units'),
                calendar=time_var.attrs.get('calendar', 'standard')
            )
            # Create a temporary dataset with decoded times for selection
            pt = pt.assign_coords(time=decoded_times)
        pt = pt.sel(time=slice(start_n, end_n))
    
    # Convert to DataFrame (xarray does this efficiently for multiple variables)
    df = pt.to_dataframe()
    
    # Handle index - depends on whether we have lat/lon in the index
    if 'lat' in df.index.names and 'lon' in df.index.names:
        # Remove spatial dimensions (they're single values)
        df = df.reset_index(level=["lat", "lon"], drop=True)
    
    # Ensure time is the index
    if 'time' not in df.index.names and 'time' in df.columns:
        df = df.set_index('time')
    
    df.index.name = "time"
    
    if dropna:
        df = df.dropna()
    
    return PointQueryResult(
        data=pt,  # xr.Dataset with all variables
        df=df,
        lat_used=float(pt["lat"].values),
        lon_used=float(pt["lon"].values),
        url=url,
    )
