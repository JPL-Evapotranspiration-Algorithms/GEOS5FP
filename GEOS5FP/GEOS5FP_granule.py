import logging
import os
from datetime import datetime
from os.path import exists, expanduser, join, splitext, basename, abspath
from typing import Any

import colored_logging as cl
import numpy as np
import rasters as rt
from rasters import RasterGeometry, Raster

 
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
            resampling = self.DEFAULT_RESAMPLING_METHOD

        if nodata is None:
            nodata = np.nan

        try:
            data = Raster.open(f'netcdf:"{self.filename_absolute}":{variable}', nodata=nodata)
        except Exception as e:
            logger.error(e)
            os.remove(self.filename)

            raise GEOS5FPGranuleNotAvailable(f"removed corrupted GEOS-5 FP file: {self.filename}")

        if exclude_values is not None:
            for exclusion_value in exclude_values:
                data = rt.where(data == exclusion_value, np.nan, data)

        data = rt.clip(data, min_value, max_value)

        if geometry is not None:
            data = data.to_geometry(geometry, resampling=resampling)

        return data
