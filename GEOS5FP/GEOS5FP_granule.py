import logging
import os
from datetime import datetime
from os.path import exists, expanduser, join, splitext, basename, abspath
from typing import Any

import colored_logging as cl
import numpy as np
import rasters as rt
from rasters import RasterGeometry, Raster
import shapely
 
from .exceptions import GEOS5FPGranuleNotAvailable

logger = logging.getLogger(__name__)

class GEOS5FPGranule:
    DEFAULT_RESAMPLING_METHOD = "cubic"

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
                resampling = self.DEFAULT_RESAMPLING_METHOD

        if nodata is None:
            nodata = np.nan

        try:
            URI = f'netcdf:"{self.filename_absolute}":{variable}'
            # print(f"reading URI: {URI} nodata: {nodata} geometry: {geometry} resampling: {resampling}")
            data = Raster.open(URI, nodata=nodata, geometry=geometry, resampling=resampling)
            # print(f"read complete: {URI}")
        except Exception as e:
            logger.error(e)
            os.remove(self.filename)

            raise GEOS5FPGranuleNotAvailable(f"removed corrupted GEOS-5 FP file: {self.filename}")

        if exclude_values is not None:
            for exclusion_value in exclude_values:
                data = rt.where(data == exclusion_value, np.nan, data)

        data = rt.clip(data, min_value, max_value)

        # if geometry is not None:
        #     # Use nearest neighbor resampling for Point geometries to avoid expensive cubic interpolation
        #     # when extracting single pixel values from large global rasters
        #     if hasattr(geometry, 'geom_type') and geometry.geom_type == 'Point':
        #         # shapely Point geometry
        #         data = data.to_geometry(geometry, resampling='nearest')
        #     elif hasattr(geometry, '__class__') and 'Point' in geometry.__class__.__name__:
        #         # rasters Point geometry
        #         data = data.to_geometry(geometry, resampling='nearest')
        #     else:
        #         data = data.to_geometry(geometry, resampling=resampling)

        return data
