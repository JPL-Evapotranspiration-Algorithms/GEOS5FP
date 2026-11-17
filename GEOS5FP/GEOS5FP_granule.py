import logging
import os
from datetime import datetime
from os.path import exists, expanduser, splitext, basename, abspath
from typing import Any

from affine import Affine
import numpy as np
import rasters as rt
from rasters import RasterGeometry, Raster, RasterGrid
import shapely

from .constants import DEFAULT_DOWNSAMPLING
from .exceptions import GEOS5FPGranuleNotAvailable

logger = logging.getLogger(__name__)


def _is_gdal_hdf5_plugin_error(error: Exception) -> bool:
    message = str(error)
    return (
        "gdal_HDF5.so" in message
        or "not recognized as being in a supported file format" in message
        or "driver HDF5" in message
    )


class GEOS5FPGranule:
    def __init__(self, filename: str):
        if not exists(abspath(expanduser(filename))):
            raise IOError(f"GEOS-5 FP file does not exist: {filename}")

        self.filename = filename

    def __repr__(self) -> str:
        return f"GEOS5FPGranule({self.filename})"

    @property
    def product(self) -> str:
        return str(splitext(basename(self.filename))[0].split(".")[-3])

    @property
    def time_UTC(self) -> datetime:
        return datetime.strptime(splitext(basename(self.filename))[0].split(".")[-2], "%Y%m%d_%H%M")

    @property
    def filename_stem(self) -> str:
        return splitext(basename(self.filename))[0]

    @property
    def filename_absolute(self) -> str:
        return abspath(expanduser(self.filename))

    def _read_via_netcdf4(
            self,
            variable: str,
            geometry: RasterGeometry = None,
            resampling: str = None,
            nodata: Any = None) -> Raster:
        import netCDF4

        with netCDF4.Dataset(self.filename_absolute, mode="r") as ds:
            if variable not in ds.variables:
                raise KeyError(f"Variable '{variable}' not found in {self.filename}")

            nc_var = ds.variables[variable]
            raw_data = nc_var[:]
            if np.ma.isMaskedArray(raw_data):
                arr = np.ma.filled(raw_data, np.nan).astype(np.float64)
            else:
                arr = np.asarray(raw_data, dtype=np.float64)

            arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D variable array for '{variable}', got shape {arr.shape}")

            # Handle GEOS-5 FP fill values (commonly 1e15)
            fill_value = getattr(nc_var, "_FillValue", None)
            if fill_value is not None and not np.isnan(fill_value):
                arr = np.where(arr == fill_value, np.nan, arr)
            missing_value = getattr(nc_var, "missing_value", None)
            if missing_value is not None and not np.isnan(missing_value):
                arr = np.where(arr == missing_value, np.nan, arr)

            lats = np.asarray(ds.variables["lat"][:], dtype=np.float64)
            lons = np.asarray(ds.variables["lon"][:], dtype=np.float64)

            # Standard rasters/GDAL orientation is North-to-South (decreasing lat)
            if lats[0] < lats[-1]:
                arr = arr[::-1, :]
                lats = lats[::-1]

            dx = float(abs(lons[1] - lons[0])) if len(lons) > 1 else 0.625
            dy = float(abs(lats[0] - lats[1])) if len(lats) > 1 else 0.5

            x_origin = float(lons[0]) - dx / 2.0
            y_origin = float(lats[0]) + dy / 2.0

            transform = Affine(dx, 0.0, x_origin, 0.0, -dy, y_origin)
            grid = RasterGrid.from_affine(
                transform,
                rows=arr.shape[0],
                cols=arr.shape[1],
                crs="EPSG:4326"
            )

            data = Raster(arr, geometry=grid, nodata=nodata)
            if geometry is not None:
                data = data.to_geometry(geometry, resampling=resampling)

            return data

    def read(
            self,
            variable: str,
            geometry: RasterGeometry = None,
            resampling: str = None,
            nodata: Any = None,
            min_value: Any = None,
            max_value: Any = None,
            exclude_values=None) -> Raster:
        if resampling is None:
            if isinstance(geometry, (rt.Point, shapely.geometry.Point)):
                resampling = "nearest"
            else:
                resampling = DEFAULT_DOWNSAMPLING
        if nodata is None:
            nodata = np.nan

        try:
            URI = f'netcdf:"{self.filename_absolute}":{variable}'
            data = Raster.open(URI, nodata=nodata, geometry=geometry, resampling=resampling)
        except Exception as e:
            if _is_gdal_hdf5_plugin_error(e):
                logger.warning(
                    f"GDAL HDF5/NetCDF plugin unavailable ({e}); falling back to netCDF4 for {self.filename}"
                )
                try:
                    data = self._read_via_netcdf4(
                        variable=variable,
                        geometry=geometry,
                        resampling=resampling,
                        nodata=nodata
                    )
                except Exception as nc_err:
                    logger.error(nc_err)
                    if exists(self.filename):
                        os.remove(self.filename)
                    raise GEOS5FPGranuleNotAvailable(
                        f"removed corrupted GEOS-5 FP file: {self.filename}"
                    ) from nc_err
            else:
                logger.error(e)
                if exists(self.filename):
                    os.remove(self.filename)
                raise GEOS5FPGranuleNotAvailable(
                    f"removed corrupted GEOS-5 FP file: {self.filename}"
                ) from e

        if exclude_values is not None:
            for exclusion_value in exclude_values:
                data = rt.where(data == exclusion_value, np.nan, data)

        data = rt.clip(data, min_value, max_value)

        return data
