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
    ds = xr.open_dataset(url, engine=engine)

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
        da = da.sel(time=slice(start_n, end_n))

    # Make tidy DataFrame
    time_index = pd.to_datetime(da["time"].values)
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
