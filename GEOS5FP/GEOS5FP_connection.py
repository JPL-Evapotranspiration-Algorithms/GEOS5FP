import json
import logging
import os
import warnings
from datetime import date, datetime, timedelta, time
from os import makedirs
from os.path import expanduser, exists, getsize
from shutil import move
from time import sleep
from typing import List, Union, Any, Tuple
import posixpath
import colored_logging as cl
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasters as rt
import requests
from tqdm.notebook import tqdm
from dateutil import parser
from rasters import Raster, RasterGeometry
from shapely.geometry import Point, MultiPoint

from .constants import *
from .exceptions import *
from .HTTP_listing import HTTP_listing
from .GEOS5FP_granule import GEOS5FPGranule
from .timer import Timer
from .downscaling import linear_downscale, bias_correct

# Optional import for point queries via OPeNDAP
try:
    from .GEOS5FP_point import query_geos5fp_point
    HAS_OPENDAP_SUPPORT = True
except ImportError:
    HAS_OPENDAP_SUPPORT = False
    query_geos5fp_point = None

logger = logging.getLogger(__name__)

class GEOS5FPConnection:
    DEFAULT_URL_BASE = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das"
    DEFAULT_TIMEOUT_SECONDS = 500
    DEFAULT_RETRIES = 3

    def __init__(
            self,
            # working_directory parameter removed
            download_directory: str = None,
            remote: str = None,
            save_products: bool = False):
        # working_directory logic removed

        # working_directory expansion logic removed

        # working_directory logging removed

        if download_directory is None:
            download_directory = DEFAULT_DOWNLOAD_DIRECTORY

        # logger.info(f"GEOS-5 FP download directory: {cl.dir(download_directory)}")

        if remote is None:
            remote = self.DEFAULT_URL_BASE

        # self.working_directory assignment removed
        self.download_directory = download_directory
        self.remote = remote
        self._listings = {}
        self.filenames = set([])
        self.save_products = save_products

    def __repr__(self):
        display_dict = {
            "URL": self.remote,
            "download_directory": self.download_directory
        }

        display_string = json.dumps(display_dict, indent=2)

        return display_string

    def _get_variable_info(self, variable_name: str) -> Tuple[str, str, str]:
        """
        Look up variable metadata from constants.
        
        :param variable_name: The name of the variable to look up
        :return: Tuple of (description, product, variable)
        :raises KeyError: If variable_name is not found in GEOS5FP_VARIABLES
        """
        if variable_name not in GEOS5FP_VARIABLES:
            raise KeyError(f"Variable '{variable_name}' not found in GEOS5FP_VARIABLES")
        
        return GEOS5FP_VARIABLES[variable_name]

    def _is_point_geometry(self, geometry) -> bool:
        """
        Check if geometry is a point or multipoint.
        
        :param geometry: Geometry to check (can be shapely Point/MultiPoint or rasters Point/MultiPoint)
        :return: True if point geometry, False otherwise
        """
        if geometry is None:
            return False
        
        # Check shapely types
        if isinstance(geometry, (Point, MultiPoint)):
            return True
        
        # Check if it's a rasters geometry with point type
        if hasattr(geometry, 'geometry') and isinstance(geometry.geometry, (Point, MultiPoint)):
            return True
        
        # Check string representation
        geom_type = str(type(geometry).__name__).lower()
        if 'point' in geom_type:
            return True
            
        return False

    def _extract_points(self, geometry) -> List[Tuple[float, float]]:
        """
        Extract (lat, lon) coordinates from point geometry.
        
        :param geometry: Point or MultiPoint geometry
        :return: List of (lat, lon) tuples
        """
        points = []
        
        # Handle rasters geometry wrapper
        if hasattr(geometry, 'geometry'):
            geom = geometry.geometry
        else:
            geom = geometry
        
        if isinstance(geom, Point):
            # Single point: (lon, lat) in shapely
            points.append((geom.y, geom.x))
        elif isinstance(geom, MultiPoint):
            # Multiple points
            for pt in geom.geoms:
                points.append((pt.y, pt.x))
        else:
            raise ValueError(f"Unsupported geometry type for point extraction: {type(geom)}")
        
        return points

    def _check_remote(self):
        logger.info(f"checking URL: {cl.URL(self.remote)}")
        response = requests.head(self.remote)
        status = response.status_code
        duration = response.elapsed.total_seconds()

        if status == 200:
            logger.info(f"remote verified with status {cl.val(200)} in " + cl.time(
                f"{duration:0.2f}") + " seconds: {cl.URL(self.remote)}")
        else:
            raise IOError(f"status: {status} URL: {self.remote}")

    @property
    def years_available(self) -> List[date]:
        listing = self.list_URL(self.remote)
        dates = sorted([date(int(item[1:]), 1, 1) for item in listing if item.startswith("Y")])

        return dates

    def year_URL(self, year: int) -> str:
        return posixpath.join(self.remote, f"Y{year:04d}") + "/"

    def is_year_available(self, year: int) -> bool:
        return requests.head(self.year_URL(year)).status_code != 404

    def months_available_in_year(self, year) -> List[date]:
        year_URL = self.year_URL(year)

        if requests.head(year_URL).status_code == 404:
            raise GEOS5FPYearNotAvailable(f"GEOS-5 FP year not available: {year_URL}")

        listing = self.list_URL(year_URL)
        dates = sorted([date(year, int(item[1:]), 1) for item in listing if item.startswith("M")])

        return dates

    def month_URL(self, year: int, month: int) -> str:
        return posixpath.join(self.remote, f"Y{year:04d}", f"M{month:02d}") + "/"

    def is_month_available(self, year: int, month: int) -> bool:
        return requests.head(self.month_URL(year, month)).status_code != 404

    def dates_available_in_month(self, year, month) -> List[date]:
        month_URL = self.month_URL(year, month)

        if requests.head(month_URL).status_code == 404:
            raise GEOS5FPMonthNotAvailable(f"GEOS-5 FP month not available: {month_URL}")

        listing = self.list_URL(month_URL)
        dates = sorted([date(year, month, int(item[1:])) for item in listing if item.startswith("D")])

        return dates

    def day_URL(self, date_UTC: Union[date, str]) -> str:
        if isinstance(date_UTC, str):
            date_UTC = parser.parse(date_UTC).date()

        year = date_UTC.year
        month = date_UTC.month
        day = date_UTC.day
        URL = posixpath.join(self.remote, f"Y{year:04d}", f"M{month:02d}", f"D{day:02d}") + "/"

        return URL

    def is_day_available(self, date_UTC: Union[date, str]) -> bool:
        return requests.head(self.day_URL(date_UTC)).status_code != 404

    @property
    def latest_date_available(self) -> date:
        date_UTC = datetime.utcnow().date()
        year = date_UTC.year
        month = date_UTC.month

        if self.is_day_available(date_UTC):
            return date_UTC

        if self.is_month_available(year, month):
            return self.dates_available_in_month(year, month)[-1]

        if self.is_year_available(year):
            return self.dates_available_in_month(year, self.months_available_in_year(year)[-1].month)[-1]

        available_year = self.years_available[-1].year
        available_month = self.months_available_in_year(available_year)[-1].month
        available_date = self.dates_available_in_month(available_year, available_month)[-1]

        return available_date

    @property
    def latest_time_available(self) -> datetime:
        retries = 3
        wait_seconds = 30

        while retries > 0:
            retries -= 1

            try:
                return self.http_listing(self.latest_date_available).sort_values(by="time_UTC").iloc[-1].time_UTC.to_pydatetime()
            except Exception as e:
                logger.warning(e)
                sleep(wait_seconds)
                continue


    def time_from_URL(self, URL: str) -> datetime:
        return datetime.strptime(URL.split(".")[-3], "%Y%m%d_%H%M")

    def list_URL(self, URL: str, timeout: float = None, retries: int = None) -> List[str]:
        if URL in self._listings:
            return self._listings[URL]
        else:
            listing = HTTP_listing(URL, timeout=timeout, retries=retries)
            self._listings[URL] = listing

            return listing

    def http_listing(
            self,
            date_UTC: Union[datetime, str],
            product_name: str = None,
            timeout: float = None,
            retries: int = None) -> pd.DataFrame:
        if timeout is None:
            timeout = self.DEFAULT_TIMEOUT_SECONDS

        if retries is None:
            retries = self.DEFAULT_RETRIES

        day_URL = self.day_URL(date_UTC)

        if requests.head(day_URL).status_code == 404:
            raise GEOS5FPDayNotAvailable(f"GEOS-5 FP day not available: {day_URL}")

        logger.info(f"listing URL: {cl.URL(day_URL)}")
        # listing = HTTP_listing(day_URL, timeout=timeout, retries=retries)
        listing = self.list_URL(day_URL, timeout=timeout, retries=retries)

        if product_name is None:
            URLs = sorted([
                posixpath.join(day_URL, filename)
                for filename
                in listing
                if filename.endswith(".nc4")
            ])
        else:
            URLs = sorted([
                posixpath.join(day_URL, filename)
                for filename
                in listing
                if product_name in filename and filename.endswith(".nc4")
            ])

        df = pd.DataFrame({"URL": URLs})
        df["time_UTC"] = df["URL"].apply(
            lambda URL: datetime.strptime(posixpath.basename(URL).split(".")[4], "%Y%m%d_%H%M"))
        df["product"] = df["URL"].apply(lambda URL: posixpath.basename(URL).split(".")[3])
        df = df[["time_UTC", "product", "URL"]]

        return df

    def generate_filenames(
            self,
            date_UTC: Union[datetime, str],
            product_name: str,
            interval: int,
            expected_hours: List[float] = None) -> pd.DataFrame:
        if isinstance(date_UTC, str):
            date_UTC = parser.parse(date_UTC).date()

        # day_URL = self.day_URL(date_UTC)
        # logger.info(f"generating URLs under: {cl.URL(day_URL)}")

        if expected_hours is None:
            if interval == 1:
                expected_hours = np.arange(0.5, 24.5, 1)
            elif interval == 3:
                expected_hours = np.arange(0.0, 24.0, 3)
            else:
                raise ValueError(f"unrecognized GEOS-5 FP interval: {interval}")

        rows = []

        expected_times = [datetime.combine(date_UTC - timedelta(days=1), time(0)) + timedelta(
            hours=float(expected_hours[-1]))] + [
                             datetime.combine(date_UTC, time(0)) + timedelta(hours=float(hour))
                             for hour
                             in expected_hours
                         ] + [datetime.combine(date_UTC + timedelta(days=1), time(0)) + timedelta(
            hours=float(expected_hours[0]))]

        for time_UTC in expected_times:
            # time_UTC = datetime.combine(date_UTC, time(0)) + timedelta(hours=float(hour))
            filename = f"GEOS.fp.asm.{product_name}.{time_UTC:%Y%m%d_%H%M}.V01.nc4"
            day_URL = self.day_URL(time_UTC.date())
            URL = posixpath.join(day_URL, filename)
            rows.append([time_UTC, URL])

        df = pd.DataFrame(rows, columns=["time_UTC", "URL"])

        return df

    def product_listing(
            self,
            date_UTC: Union[datetime, str],
            product_name: str,
            interval: int,
            expected_hours: List[float] = None,
            timeout: float = None,
            retries: int = None,
            use_http_listing: bool = False) -> pd.DataFrame:
        if use_http_listing:
            return self.http_listing(
                date_UTC=date_UTC,
                product_name=product_name,
                timeout=timeout,
                retries=retries
            )
        elif expected_hours is not None or interval is not None:
            return self.generate_filenames(
                date_UTC=date_UTC,
                product_name=product_name,
                interval=interval,
                expected_hours=expected_hours
            )
        else:
            raise ValueError("must use HTTP listing if not supplying expected hours")

    def date_download_directory(self, time_UTC: datetime) -> str:
        return join(self.download_directory, f"{time_UTC:%Y.%m.%d}")

    def download_filename(self, URL: str) -> str:
        time_UTC = self.time_from_URL(URL)
        download_directory = self.date_download_directory(time_UTC)
        filename = join(download_directory, posixpath.basename(URL))

        return filename

    def download_file(self, URL: str, filename: str = None, retries: int = 3, wait_seconds: int = 30) -> GEOS5FPGranule:
        if filename is None:
            filename = self.download_filename(URL)

        expanded_filename = os.path.expanduser(filename)

        if exists(expanded_filename) and getsize(expanded_filename) == 0:
            logger.warning(f"removing previously created zero-size corrupted GEOS-5 FP file: {filename}")
            os.remove(expanded_filename)

        if exists(expanded_filename):
            return GEOS5FPGranule(filename)

        import requests
        from requests.exceptions import ChunkedEncodingError, ConnectionError

        while retries > 0:
            retries -= 1
            try:
                if exists(expanded_filename):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            with rasterio.open(expanded_filename, "r") as file:
                                pass
                    except Exception as e:
                        logger.exception(f"unable to open GEOS-5 FP file: {filename}")
                        logger.warning(f"removing corrupted GEOS-5 FP file: {filename}")
                        os.remove(expanded_filename)

                if exists(expanded_filename):
                    logger.info(f"GEOS-5 FP file found: {cl.file(filename)}")
                else:
                    # Verify that the file exists at the remote
                    if requests.head(URL).status_code == 404:
                        directory_URL = posixpath.dirname(URL)
                        if requests.head(directory_URL).status_code == 404:
                            raise GEOS5FPDayNotAvailable(f"GEOS-5 FP day not available: {directory_URL}")
                        else:
                            raise GEOS5FPGranuleNotAvailable(f"GEOS-5 FP granule not available: {URL}")

                    logger.info(f"downloading GEOS-5 FP: {cl.URL(URL)} -> {cl.file(filename)}")
                    makedirs(os.path.dirname(expanded_filename), exist_ok=True)
                    partial_filename = f"{filename}.{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.download"
                    expanded_partial_filename = os.path.expanduser(partial_filename)

                    if exists(expanded_partial_filename) and getsize(expanded_partial_filename) == 0:
                        logger.warning(f"removing zero-size corrupted GEOS-5 FP file: {partial_filename}")
                        os.remove(expanded_partial_filename)

                    # Download with requests and TQDM progress bar
                    timer = Timer()
                    logger.info(f"downloading with requests: {URL} -> {expanded_partial_filename}")
                    try:
                        response = requests.get(URL, stream=True, timeout=120)
                        response.raise_for_status()
                        total = int(response.headers.get('content-length', 0))
                        with open(expanded_partial_filename, 'wb') as f, tqdm(
                            desc=posixpath.basename(expanded_partial_filename),
                            total=total,
                            unit='B',
                            unit_scale=True,
                            unit_divisor=1024,
                            leave=True
                        ) as bar:
                            for chunk in response.iter_content(chunk_size=1024*1024):
                                if chunk:
                                    f.write(chunk)
                                    bar.update(len(chunk))
                    except (ChunkedEncodingError, ConnectionError) as e:
                        logger.error(f"Network error during download: {e}")
                        if exists(expanded_partial_filename):
                            os.remove(expanded_partial_filename)
                        if retries == 0:
                            raise FailedGEOS5FPDownload(f"requests download failed: {URL} -> {partial_filename}")
                        logger.warning(f"waiting {wait_seconds} seconds for retry")
                        sleep(wait_seconds)
                        continue
                    except Exception as e:
                        logger.exception(f"Download failed: {e}")
                        if exists(expanded_partial_filename):
                            os.remove(expanded_partial_filename)
                        raise FailedGEOS5FPDownload(f"requests download failed: {URL} -> {partial_filename}")

                    if not exists(expanded_partial_filename):
                        raise IOError(f"unable to download URL: {URL}")

                    if exists(expanded_partial_filename) and getsize(expanded_partial_filename) == 0:
                        logger.warning(f"removing zero-size corrupted GEOS-5 FP file: {partial_filename}")
                        os.remove(expanded_partial_filename)
                        raise FailedGEOS5FPDownload(f"zero-size file from GEOS-5 FP download: {URL} -> {partial_filename}")

                    move(expanded_partial_filename, expanded_filename)

                    try:
                        with rasterio.open(expanded_filename, "r") as file:
                            pass
                    except Exception as e:
                        logger.exception(f"unable to open GEOS-5 FP file: {filename}")
                        logger.warning(f"removing corrupted GEOS-5 FP file: {filename}")
                        os.remove(expanded_filename)
                        raise FailedGEOS5FPDownload(f"GEOS-5 FP corrupted download: {URL} -> {filename}")

                    logger.info(f"GEOS-5 FP download completed: {cl.file(filename)} ({(getsize(expanded_filename) / 1000000):0.2f} MB) ({cl.time(timer.duration)} seconds)")

                granule = GEOS5FPGranule(filename)

                return granule

            except Exception as e:
                if retries == 0:
                    raise e

                logger.warning(e)
                logger.warning(f"waiting {wait_seconds} seconds for retry")
                sleep(wait_seconds)
                continue

    def before_and_after(
            self,
            time_UTC: Union[datetime, str],
            product: str,
            interval: int = None,
            expected_hours: List[float] = None,
            timeout: float = None,
            retries: int = None,
            use_http_listing: bool = DEFAULT_USE_HTTP_LISTING) -> Tuple[datetime, Raster, datetime, Raster]:
        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)

        ## FIXME need to check local filesystem for existing files here first before searching remote

        search_date = time_UTC.date()
        logger.info(f"searching GEOS-5 FP {cl.name(product)} at " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M:%S} UTC"))

        product_listing = self.product_listing(
            search_date,
            product,
            interval=interval,
            expected_hours=expected_hours,
            timeout=timeout,
            retries=retries,
            use_http_listing=use_http_listing
        )

        if len(product_listing) == 0:
            raise IOError(f"no {product} files found for {time_UTC}")

        before_listing = product_listing[product_listing.time_UTC < time_UTC]

        if len(before_listing) == 0:
            raise IOError(f"no {product} files found preceeding {time_UTC}")

        before_time_UTC, before_URL = before_listing.iloc[-1][["time_UTC", "URL"]]
        after_listing = product_listing[product_listing.time_UTC > time_UTC]

        if len(after_listing) == 0:
            after_listing = self.product_listing(
                search_date + timedelta(days=1),
                product,
                interval=interval,
                expected_hours=expected_hours,
                timeout=timeout,
                retries=retries,
                use_http_listing=use_http_listing
            )
            # raise IOError(f"no {product} files found after {time_UTC}")

        after_time_UTC, after_URL = after_listing.iloc[0][["time_UTC", "URL"]]
        before_granule = self.download_file(before_URL)
        after_granule = self.download_file(after_URL)

        return before_granule, after_granule

    def _get_simple_variable(
            self,
            variable_name: str,
            time_UTC: Union[datetime, str],
            geometry: RasterGeometry = None,
            resampling: str = None,
            interval: int = None,
            expected_hours: List[float] = None,
            min_value: Any = None,
            max_value: Any = None,
            exclude_values=None,
            cmap=None,
            clip_min: Any = None,
            clip_max: Any = None) -> Union[Raster, pd.DataFrame]:
        """
        Generic method to retrieve a simple variable that requires only interpolation.
        
        :param variable_name: Name of the variable (must exist in GEOS5FP_VARIABLES)
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry (can be raster geometry or Point/MultiPoint for time-series)
        :param resampling: optional sampling method for resampling to target geometry
        :param interval: optional interval for product listing
        :param expected_hours: optional expected hours for product listing
        :param min_value: minimum value for interpolation
        :param max_value: maximum value for interpolation
        :param exclude_values: values to exclude in interpolation
        :param cmap: colormap for the result
        :param clip_min: minimum value for final clipping (applied after interpolation)
        :param clip_max: maximum value for final clipping (applied after interpolation)
        :return: raster of the requested variable or DataFrame for point queries or DataFrame for point queries
        """
        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)
            
        NAME, PRODUCT, VARIABLE = self._get_variable_info(variable_name)
        
        # Check if this is a point query
        if self._is_point_geometry(geometry):
            if not HAS_OPENDAP_SUPPORT:
                raise ImportError(
                    "Point query support requires xarray and netCDF4. "
                    "Install with: conda install -c conda-forge xarray netcdf4"
                )
            
            logger.info(
                f"retrieving {cl.name(NAME)} time-series "
                f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
                "at point location(s)"
            )
            
            points = self._extract_points(geometry)
            dfs = []
            
            # OPeNDAP uses lowercase variable names
            variable_opendap = VARIABLE.lower()
            
            for lat, lon in points:
                try:
                    # For single time point, query a small window around the time
                    # to ensure we get the nearest available timestep
                    time_window_start = time_UTC - timedelta(hours=2)
                    time_window_end = time_UTC + timedelta(hours=2)
                    
                    result = query_geos5fp_point(
                        dataset=PRODUCT,
                        variable=variable_opendap,
                        lat=lat,
                        lon=lon,
                        time_range=(time_window_start, time_window_end),
                        dropna=True
                    )
                    
                    if len(result.df) == 0:
                        logger.warning(f"No data found for point ({lat}, {lon}) near {time_UTC}")
                        continue
                    
                    # Find the closest time to our target
                    time_diffs = abs(result.df.index - time_UTC)
                    closest_idx = time_diffs.argmin()
                    df_point = result.df.iloc[[closest_idx]].copy()
                    
                    df_point['lat'] = lat
                    df_point['lon'] = lon
                    df_point['lat_used'] = result.lat_used
                    df_point['lon_used'] = result.lon_used
                    dfs.append(df_point)
                except Exception as e:
                    logger.warning(f"Failed to query point ({lat}, {lon}): {e}")
                    
            if not dfs:
                raise ValueError("No successful point queries")
            
            result_df = pd.concat(dfs, ignore_index=False)
            # Rename from OPeNDAP variable name to our standard variable name
            result_df = result_df.rename(columns={variable_opendap: variable_name})
            
            # Apply clipping if specified
            if clip_min is not None or clip_max is not None:
                result_df[variable_name] = result_df[variable_name].clip(lower=clip_min, upper=clip_max)
            
            return result_df
        
        # Regular raster query
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )
        
        result = self.interpolate(
            time_UTC=time_UTC,
            product=PRODUCT,
            variable=VARIABLE,
            geometry=geometry,
            resampling=resampling,
            interval=interval,
            expected_hours=expected_hours,
            min_value=min_value,
            max_value=max_value,
            exclude_values=exclude_values,
            cmap=cmap
        )
        
        if clip_min is not None or clip_max is not None:
            result = rt.clip(result, clip_min, clip_max)
            
        return result

    def interpolate(
            self,
            time_UTC: Union[datetime, str],
            product: str,
            variable: str,
            geometry: RasterGeometry = None,
            resampling: str = None,
            cmap=None,
            min_value: Any = None,
            max_value: Any = None,
            exclude_values=None,
            interval: int = None,
            expected_hours: List[float] = None,
            timeout: float = None,
            retries: int = None,
            use_http_listing: bool = DEFAULT_USE_HTTP_LISTING) -> Raster:
        if interval is None:
            if product == "tavg1_2d_rad_Nx":
                interval = 1
            elif product == "tavg1_2d_slv_Nx":
                interval = 1
            elif product == "inst3_2d_asm_Nx":
                interval = 3

        if interval is None and expected_hours is None:
            raise ValueError(f"interval or expected hours not given for {product}")

        before_granule, after_granule = self.before_and_after(
            time_UTC,
            product,
            interval=interval,
            expected_hours=expected_hours,
            timeout=timeout,
            retries=retries,
            use_http_listing=use_http_listing
        )

        logger.info(
            f"interpolating GEOS-5 FP {cl.name(product)} {cl.name(variable)} " +
            f"from " + cl.time(f"{before_granule.time_UTC:%Y-%m-%d %H:%M} UTC ") +
            f"and " + cl.time(f"{after_granule.time_UTC:%Y-%m-%d %H:%M} UTC") + " to " + cl.time(
                f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        with Timer() as timer:
            before = before_granule.read(
                variable,
                geometry=geometry,
                resampling=resampling,
                min_value=min_value,
                max_value=max_value,
                exclude_values=exclude_values
            )

            after = after_granule.read(
                variable,
                geometry=geometry,
                resampling=resampling,
                min_value=min_value,
                max_value=max_value,
                exclude_values=exclude_values
            )

            time_fraction = (time_UTC - before_granule.time_UTC) / (after_granule.time_UTC - before_granule.time_UTC)
            source_diff = after - before
            interpolated_data = before + source_diff * time_fraction
            logger.info(f"GEOS-5 FP interpolation complete ({timer:0.2f} seconds)")

        before_filename = before_granule.filename
        after_filename = after_granule.filename
        filenames = [before_filename, after_filename]
        self.filenames = set(self.filenames) | set(filenames)
        interpolated_data["filenames"] = filenames

        if cmap is not None:
            interpolated_data.cmap = cmap

        return interpolated_data


    def SFMC(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        top soil layer moisture content cubic meters per cubic meters
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry (raster geometry or Point/MultiPoint for time-series)
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of soil moisture or DataFrame for point queries
        """
        return self._get_simple_variable(
            "SFMC",
            time_UTC,
            geometry=geometry,
            resampling=resampling,
            interval=1,
            min_value=0,
            max_value=1,
            exclude_values=[1],
            cmap=SM_CMAP
        )

    SM = SFMC

    def LAI(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        leaf area index
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of LAI or DataFrame for point queries
        """
        return self._get_simple_variable(
            "LAI",
            time_UTC,
            geometry=geometry,
            resampling=resampling,
            interval=1,
            min_value=0,
            max_value=10,
            cmap=NDVI_CMAP
        )

    def NDVI(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        normalized difference vegetation index
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of NDVI or DataFrame for point queries
        """
        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)
        LAI = self.LAI(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        NDVI = rt.clip(1.05 - np.exp(-0.5 * LAI), 0, 1)

        return NDVI

    def LHLAND(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        latent heat flux in watts per square meter
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of soil moisture or DataFrame for point queries or DataFrame for point queries
        """
        return self._get_simple_variable(
            "LHLAND",
            time_UTC,
            geometry=geometry,
            resampling=resampling,
            interval=1,
            exclude_values=[1.e+15]
        )

    def EFLUX(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        latent heat flux in watts per square meter
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of soil moisture or DataFrame for point queries or DataFrame for point queries
        """
        return self._get_simple_variable(
            "EFLUX",
            time_UTC,
            geometry=geometry,
            resampling=resampling,
            interval=1
        )

    def PARDR(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        Surface downward PAR beam flux in watts per square meter
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of soil moisture or DataFrame for point queries or DataFrame for point queries
        """
        return self._get_simple_variable(
            "PARDR",
            time_UTC,
            geometry=geometry,
            resampling=resampling,
            clip_min=0
        )

    def PARDF(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        Surface downward PAR diffuse flux in watts per square meter
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of soil moisture or DataFrame for point queries or DataFrame for point queries
        """
        return self._get_simple_variable(
            "PARDF",
            time_UTC,
            geometry=geometry,
            resampling=resampling,
            clip_min=0
        )

    def AOT(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        aerosol optical thickness (AOT) extinction coefficient
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of AOT or DataFrame for point queries or DataFrame for point queries
        """
        # 1:30, 4:30, 7:30, 10:30, 13:30, 16:30, 19:30, 22:30 UTC
        EXPECTED_HOURS = [1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5]
        
        return self._get_simple_variable(
            "AOT",
            time_UTC,
            geometry=geometry,
            resampling=resampling,
            expected_hours=EXPECTED_HOURS
        )

    def COT(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        cloud optical thickness (COT) extinction coefficient
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of COT or DataFrame for point queries or DataFrame for point queries
        """
        return self._get_simple_variable(
            "COT",
            time_UTC,
            geometry=geometry,
            resampling=resampling
        )

    def Ts_K(
            self,
            time_UTC: Union[datetime, str],
            geometry: RasterGeometry = None,
            resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        surface temperature (Ts) in Kelvin
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of Ta or DataFrame for point queries or DataFrame for point queries
        """
        return self._get_simple_variable(
            "Ts_K",
            time_UTC,
            geometry=geometry,
            resampling=resampling
        )

    def Ta_K(
            self,
            time_UTC: Union[datetime, str],
            geometry: RasterGeometry = None,
            ST_K: Raster = None,
            water: Raster = None,
            coarse_geometry: RasterGeometry = None,
            coarse_cell_size_meters: int = DEFAULT_COARSE_CELL_SIZE_METERS,
            resampling: str = None,
            upsampling: str = None,
            downsampling: str = None,
            apply_scale: bool = True,
            apply_bias: bool = True,
            return_scale_and_bias: bool = False) -> Union[Raster, pd.DataFrame]:
        """
        near-surface air temperature (Ta) in Kelvin
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry (raster or Point/MultiPoint)
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of Ta or DataFrame for point queries
        """
        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)
        
        # If point geometry and no downscaling requested, use simple variable retrieval
        if self._is_point_geometry(geometry) and ST_K is None:
            return self._get_simple_variable(
                "Ta_K",
                time_UTC,
                geometry=geometry,
                resampling=resampling
            )
        
        NAME, PRODUCT, VARIABLE = self._get_variable_info("Ta_K")

        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        if coarse_cell_size_meters is None:
            coarse_cell_size_meters = DEFAULT_COARSE_CELL_SIZE_METERS

        if ST_K is None:
            return self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)
        else:
            if geometry is None:
                geometry = ST_K.geometry

            if coarse_geometry is None:
                coarse_geometry = geometry.rescale(coarse_cell_size_meters)

            Ta_K_coarse = self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=coarse_geometry, resampling=resampling)
            filenames = Ta_K_coarse["filenames"]

            ST_K_water = None

            if water is not None:
                ST_K_water = rt.where(water, ST_K, np.nan)
                ST_K = rt.where(water, np.nan, ST_K)

            scale = None
            bias = None

            Ta_K = linear_downscale(
                coarse_image=Ta_K_coarse,
                fine_image=ST_K,
                upsampling=upsampling,
                downsampling=downsampling,
                apply_scale=apply_scale,
                apply_bias=apply_bias,
                return_scale_and_bias=return_scale_and_bias
            )

            if water is not None:
                # Ta_K_smooth = self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling="linear")
                Ta_K_water = linear_downscale(
                    coarse_image=Ta_K_coarse,
                    fine_image=ST_K_water,
                    upsampling=upsampling,
                    downsampling=downsampling,
                    apply_scale=apply_scale,
                    apply_bias=apply_bias,
                    return_scale_and_bias=False
                )

                Ta_K = rt.where(water, Ta_K_water, Ta_K)

            Ta_K.filenames = filenames

            return Ta_K

    def Tmin_K(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        minimum near-surface air temperature (Ta) in Kelvin
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of Ta or DataFrame for point queries
        """
        return self._get_simple_variable(
            "Tmin_K",
            time_UTC,
            geometry=geometry,
            resampling=resampling
        )

    def SVP_Pa(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        Ta_C = self.Ta_C(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        SVP_Pa = 0.6108 * np.exp((17.27 * Ta_C) / (Ta_C + 237.3)) * 1000  # [Pa]

        return SVP_Pa

    def SVP_kPa(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        return self.SVP_Pa(time_UTC=time_UTC, geometry=geometry, resampling=resampling) / 1000

    def Ta_C(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        return self.Ta_K(time_UTC=time_UTC, geometry=geometry, resampling=resampling) - 273.15

    def PS(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        surface pressure in Pascal
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of surface pressure or DataFrame for point queries or DataFrame for point queries
        """
        return self._get_simple_variable(
            "PS",
            time_UTC,
            geometry=geometry,
            resampling=resampling
        )

    def Q(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        near-surface specific humidity (Q) in kilograms per kilogram
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of Q or DataFrame for point queries or DataFrame for point queries
        """
        return self._get_simple_variable(
            "Q",
            time_UTC,
            geometry=geometry,
            resampling=resampling
        )

    def Ea_Pa(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        RH = self.RH(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        SVP_Pa = self.SVP_Pa(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        Ea_Pa = RH * SVP_Pa

        return Ea_Pa

    def Td_K(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        Ta_K = self.Ta_K(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        RH = self.RH(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        Td_K = Ta_K - (100 - (RH * 100)) / 5

        return Td_K

    def Td_C(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        return self.Td_K(time_UTC=time_UTC, geometry=geometry, resampling=resampling) - 273.15

    def Cp(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        Ps_Pa = self.PS(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        Ea_Pa = self.Ea_Pa(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        Cp = 0.24 * 4185.5 * (1.0 + 0.8 * (0.622 * Ea_Pa / (Ps_Pa - Ea_Pa)))  # [J kg-1 K-1]

        return Cp

    def VPD_Pa(
            self,
            time_UTC: Union[datetime, str],
            ST_K: Raster = None,
            geometry: RasterGeometry = None,
            coarse_geometry: RasterGeometry = None,
            coarse_cell_size_meters: int = DEFAULT_COARSE_CELL_SIZE_METERS,
            resampling: str = None,
            upsampling: str = None,
            downsampling: str = None,
            return_scale_and_bias: bool = False) -> Raster:
        if ST_K is None:
            Ea_Pa = self.Ea_Pa(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
            SVP_Pa = self.SVP_Pa(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
            VPD_Pa = rt.clip(SVP_Pa - Ea_Pa, 0, None)

            return VPD_Pa
        else:
            if geometry is None:
                geometry = ST_K.geometry

            if coarse_geometry is None:
                coarse_geometry = geometry.rescale(coarse_cell_size_meters)

            Ea_Pa = self.Ea_Pa(time_UTC=time_UTC, geometry=coarse_geometry, resampling=resampling)
            SVP_Pa = self.SVP_Pa(time_UTC=time_UTC, geometry=coarse_geometry, resampling=resampling)
            VPD_Pa = rt.clip(SVP_Pa - Ea_Pa, 0, None)

            return linear_downscale(
                coarse_image=VPD_Pa,
                fine_image=ST_K,
                upsampling=upsampling,
                downsampling=downsampling,
                return_scale_and_bias=return_scale_and_bias
            )

    def VPD_kPa(
            self,
            time_UTC: Union[datetime, str],
            ST_K: Raster = None,
            geometry: RasterGeometry = None,
            coarse_geometry: RasterGeometry = None,
            coarse_cell_size_meters: int = DEFAULT_COARSE_CELL_SIZE_METERS,
            resampling: str = None,
            upsampling: str = None,
            downsampling: str = None) -> Raster:
        VPD_Pa = self.VPD_Pa(
            time_UTC=time_UTC,
            ST_K=ST_K,
            geometry=geometry,
            coarse_geometry=coarse_geometry,
            coarse_cell_size_meters=coarse_cell_size_meters,
            resampling=resampling,
            upsampling=upsampling,
            downsampling=downsampling
        )

        VPD_kPa = VPD_Pa / 1000

        return VPD_kPa

    def RH(
            self,
            time_UTC: Union[datetime, str],
            geometry: RasterGeometry = None,
            SM: Raster = None,
            ST_K: Raster = None,
            VPD_kPa: Raster = None,
            water: Raster = None,
            coarse_geometry: RasterGeometry = None,
            coarse_cell_size_meters: int = DEFAULT_COARSE_CELL_SIZE_METERS,
            resampling: str = None,
            upsampling: str = None,
            downsampling: str = None,
            sharpen_VPD: bool = True,
            return_bias: bool = False) -> Raster:
        if upsampling is None:
            upsampling = DEFAULT_UPSAMPLING

        if downsampling is None:
            downsampling = DEFAULT_DOWNSAMPLING

        bias_fine = None

        if SM is None:
            Q = self.Q(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
            Ps_Pa = self.PS(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
            SVP_Pa = self.SVP_Pa(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
            Mw = 18.015268  # g / mol
            Md = 28.96546e-3  # kg / mol
            epsilon = Mw / (Md * 1000)
            w = Q / (1 - Q)
            ws = epsilon * SVP_Pa / (Ps_Pa - SVP_Pa)
            RH = rt.clip(w / ws, 0, 1)
        else:
            if geometry is None:
                geometry = SM.geometry

            if coarse_geometry is None:
                coarse_geometry = geometry.rescale(coarse_cell_size_meters)

            RH_coarse = self.RH(time_UTC=time_UTC, geometry=coarse_geometry, resampling=resampling)

            if VPD_kPa is None:
                if sharpen_VPD:
                    VPD_fine_distribution = ST_K
                else:
                    VPD_fine_distribution = None

                VPD_kPa = self.VPD_kPa(
                    time_UTC=time_UTC,
                    ST_K=VPD_fine_distribution,
                    geometry=geometry,
                    coarse_geometry=coarse_geometry,
                    coarse_cell_size_meters=coarse_cell_size_meters,
                    resampling=resampling,
                    upsampling=upsampling,
                    downsampling=downsampling
                )

            RH_estimate_fine = SM ** (1 / VPD_kPa)

            if return_bias:
                RH, bias_fine = bias_correct(
                    coarse_image=RH_coarse,
                    fine_image=RH_estimate_fine,
                    upsampling=upsampling,
                    downsampling=downsampling,
                    return_bias=True
                )
            else:
                RH = bias_correct(
                    coarse_image=RH_coarse,
                    fine_image=RH_estimate_fine,
                    upsampling=upsampling,
                    downsampling=downsampling,
                    return_bias=False
                )

            if water is not None:
                if ST_K is not None:
                    ST_K_water = rt.where(water, ST_K, np.nan)
                    RH_coarse_complement = 1 - RH_coarse
                    RH_complement_water = linear_downscale(
                        coarse_image=RH_coarse_complement,
                        fine_image=ST_K_water,
                        upsampling=upsampling,
                        downsampling=downsampling,
                        apply_bias=True,
                        return_scale_and_bias=False
                    )

                    RH_water = 1 - RH_complement_water
                    RH = rt.where(water, RH_water, RH)
                else:
                    RH_smooth = self.RH(time_UTC=time_UTC, geometry=geometry, resampling="linear")
                    RH = rt.where(water, RH_smooth, RH)

        RH = rt.clip(RH, 0, 1)

        if return_bias:
            return RH, bias_fine
        else:
            return RH

    def Ea_kPa(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        return self.Ea_Pa(time_UTC=time_UTC, geometry=geometry, resampling=resampling) / 1000

    def vapor_kgsqm(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        total column water vapor (vapor_gccm) in kilograms per square meter
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of vapor_gccm or DataFrame for point queries or DataFrame for point queries or DataFrame for point queries
        """
        return self._get_simple_variable(
            "vapor_kgsqm",
            time_UTC,
            geometry=geometry,
            resampling=resampling,
            clip_min=0
        )

    def vapor_gccm(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        total column water vapor (vapor_gccm) in grams per square centimeter
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of vapor_gccm or DataFrame for point queries
        """
        return self.vapor_kgsqm(time_UTC=time_UTC, geometry=geometry, resampling=resampling) / 10

    def ozone_dobson(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        total column ozone (ozone_cm) in Dobson units
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of vapor_gccm or DataFrame for point queries or DataFrame for point queries
        """
        return self._get_simple_variable(
            "ozone_dobson",
            time_UTC,
            geometry=geometry,
            resampling=resampling,
            clip_min=0
        )

    def ozone_cm(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        total column ozone (ozone_cm) in centimeters
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of vapor_gccm or DataFrame for point queries
        """
        return self.ozone_dobson(time_UTC=time_UTC, geometry=geometry, resampling=resampling) / 1000

    def U2M(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        eastward wind at 2 meters in meters per second
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of vapor_gccm or DataFrame for point queries or DataFrame for point queries
        """
        return self._get_simple_variable(
            "U2M",
            time_UTC,
            geometry=geometry,
            resampling=resampling
        )

    def V2M(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        northward wind at 2 meters in meters per second
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of vapor_gccm or DataFrame for point queries or DataFrame for point queries
        """
        return self._get_simple_variable(
            "V2M",
            time_UTC,
            geometry=geometry,
            resampling=resampling
        )

    def CO2SC(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        carbon dioxide suface concentration in ppm or micromol per mol
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of vapor_gccm or DataFrame for point queries or DataFrame for point queries
        """
        # 1:30, 4:30, 7:30, 10:30, 13:30, 16:30, 19:30, 22:30 UTC
        EXPECTED_HOURS = [1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5]
        
        return self._get_simple_variable(
            "CO2SC",
            time_UTC,
            geometry=geometry,
            resampling=resampling,
            expected_hours=EXPECTED_HOURS
        )

    def wind_speed(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        wind speed in meters per second
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of vapor_gccm or DataFrame for point queries
        """
        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)
        U = self.U2M(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        V = self.V2M(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        wind_speed = rt.clip(np.sqrt(U ** 2.0 + V ** 2.0), 0.0, None)

        return wind_speed

    def SWin(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        incoming shortwave radiation (SWin) in watts per square meter
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of SWin or DataFrame for point queries or DataFrame for point queries or DataFrame for point queries
        """
        return self._get_simple_variable(
            "SWin",
            time_UTC,
            geometry=geometry,
            resampling=resampling,
            clip_min=0
        )

    def SWTDN(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        top of atmosphere incoming shortwave radiation (SWin) in watts per square meter
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of SWin or DataFrame for point queries or DataFrame for point queries
        """
        return self._get_simple_variable(
            "SWTDN",
            time_UTC,
            geometry=geometry,
            resampling=resampling,
            clip_min=0
        )

    def ALBVISDR(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        Direct beam VIS-UV surface albedo
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of direct visible albedo or DataFrame for point queries or DataFrame for point queries
        """
        return self._get_simple_variable(
            "ALBVISDR",
            time_UTC,
            geometry=geometry,
            resampling=resampling,
            clip_min=0,
            clip_max=1
        )

    def ALBVISDF(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        Diffuse beam VIS-UV surface albedo
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of direct visible albedo or DataFrame for point queries or DataFrame for point queries
        """
        return self._get_simple_variable(
            "ALBVISDF",
            time_UTC,
            geometry=geometry,
            resampling=resampling,
            clip_min=0,
            clip_max=1
        )

    def ALBNIRDF(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        Diffuse beam NIR surface albedo
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of direct visible albedo or DataFrame for point queries or DataFrame for point queries
        """
        return self._get_simple_variable(
            "ALBNIRDF",
            time_UTC,
            geometry=geometry,
            resampling=resampling,
            clip_min=0,
            clip_max=1
        )

    def ALBNIRDR(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        Direct beam NIR surface albedo
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of direct visible albedo or DataFrame for point queries or DataFrame for point queries
        """
        return self._get_simple_variable(
            "ALBNIRDR",
            time_UTC,
            geometry=geometry,
            resampling=resampling,
            clip_min=0,
            clip_max=1
        )

    def ALBEDO(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        Surface albedo
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of direct visible albedo or DataFrame for point queries or DataFrame for point queries
        """
        return self._get_simple_variable(
            "ALBEDO",
            time_UTC,
            geometry=geometry,
            resampling=resampling,
            clip_min=0,
            clip_max=1
        )

    def variable(
            self,
            variable_name: Union[str, List[str]],
            time_UTC: Union[datetime, str, List[datetime], List[str], pd.Series] = None,
            time_range: Tuple[Union[datetime, str], Union[datetime, str]] = None,
            dataset: str = None,
            geometry: Union[RasterGeometry, Point, MultiPoint, List, gpd.GeoSeries] = None,
            resampling: str = None,
            lat: Union[float, List[float], pd.Series] = None,
            lon: Union[float, List[float], pd.Series] = None,
            dropna: bool = True,
            **kwargs) -> Union[Raster, gpd.GeoDataFrame]:
        """
        General-purpose variable retrieval method that can query any variable from any dataset.
        
        This method provides a flexible interface for retrieving GEOS-5 FP data, supporting both:
        - Raster queries (for spatial data over a region)
        - Point queries (for time-series at specific coordinates using OPeNDAP)
        - Batch spatio-temporal queries (multiple points at different times)
        
        Parameters
        ----------
        variable_name : str or list of str
            Name(s) of the variable(s) to retrieve. Can be either:
            - A single variable name (str): "Ta_K", "SM", "SWin", etc.
            - A list of variable names (List[str]): ["Ta_K", "SM", "SWin"]
            Each can be either:
            - A predefined variable name from GEOS5FP_VARIABLES (e.g., "Ta_K", "SM", "SWin")
            - A raw GEOS-5 FP variable name (e.g., "T2M", "SFMC", "SWGDN")
            When multiple variables are requested with point geometry, each variable becomes a column.
        time_UTC : datetime, str, list, or Series, optional
            Date/time in UTC. Can be:
            - Single datetime or str for raster queries or single point query
            - List of datetimes or Series for batch spatio-temporal queries
            Required if time_range not provided.
        time_range : tuple of (start, end), optional
            Time range (start_time, end_time) for time-series point queries.
            Use this with lat/lon for efficient multi-timestep queries at same location.
        dataset : str, optional
            GEOS-5 FP product/dataset name (e.g., "tavg1_2d_slv_Nx", "inst3_2d_asm_Nx").
            If not provided, will be looked up from GEOS5FP_VARIABLES.
            Required when using raw variable names or when querying multiple variables from same dataset.
        geometry : RasterGeometry or Point or MultiPoint, optional
            Target geometry. Can be:
            - RasterGeometry for spatial raster queries
            - shapely Point or MultiPoint for time-series queries
            - None for native GEOS-5 FP resolution
        resampling : str, optional
            Resampling method when reprojecting to target geometry (e.g., "nearest", "bilinear")
        lat : float, optional
            Latitude for point query (alternative to providing Point geometry)
        lon : float, optional
            Longitude for point query (alternative to providing Point geometry)
        dropna : bool, default=True
            Whether to drop NaN values from point query results
        **kwargs : dict
            Additional keyword arguments passed to interpolation or query functions
        
        Returns
        -------
        Raster or pd.DataFrame
            - Raster if querying spatial data at a single time (single variable only)
            - DataFrame if querying point location(s), with time index and variable column(s)
        
        Examples
        --------
        # Example 1: Single-point time-series using OPeNDAP (fast!)
        >>> from datetime import datetime, timedelta
        >>> conn = GEOS5FPConnection()
        >>> end_time = datetime(2024, 11, 15)
        >>> start_time = end_time - timedelta(days=7)
        >>> df = conn.variable(
        ...     "T2M",
        ...     time_range=(start_time, end_time),
        ...     dataset="tavg1_2d_slv_Nx",
        ...     lat=34.05,
        ...     lon=-118.25
        ... )
        
        # Example 2: Multiple variables in point query
        >>> df = conn.variable(
        ...     ["T2M", "PS", "QV2M"],
        ...     time_range=(start_time, end_time),
        ...     dataset="tavg1_2d_slv_Nx",
        ...     lat=34.05,
        ...     lon=-118.25
        ... )
        
        # Example 3: Raster query at single time
        >>> from rasters import RasterGeometry
        >>> geometry = RasterGeometry.open("target_area.tif")
        >>> raster = conn.variable(
        ...     "T2M",
        ...     time_UTC="2024-11-15 12:00:00",
        ...     dataset="tavg1_2d_slv_Nx",
        ...     geometry=geometry
        ... )
        
        # Example 4: Use predefined variable name
        >>> df = conn.variable(
        ...     "Ta_K",
        ...     time_range=(start_time, end_time),
        ...     lat=34.05,
        ...     lon=-118.25
        ... )
        
        # Example 5: Query multiple predefined variables
        >>> df = conn.variable(
        ...     ["Ta_K", "SM", "SWin"],
        ...     time_range=(start_time, end_time),
        ...     lat=40.7,
        ...     lon=-74.0
        ... )
        
        Notes
        -----
        - For time-series point queries, this method uses OPeNDAP which is much faster
          than iterating through individual timesteps
        - When both time_UTC and time_range are provided, time_range takes precedence
          for point queries
        - Point queries require xarray and netCDF4 to be installed
        - Multiple variables can only be queried simultaneously for point geometries
        - Raster queries only support single variables
        """
        # Normalize variable_name to list
        if isinstance(variable_name, str):
            variable_names = [variable_name]
            single_variable = True
        else:
            variable_names = variable_name
            single_variable = False
        
        # Validate inputs
        if time_UTC is None and time_range is None:
            raise ValueError("Either time_UTC or time_range must be provided")
        
        # Check for vectorized batch query (lists of times and geometries)
        is_batch_query = False
        if time_UTC is not None and time_range is None:
            # Check if time_UTC is a list/Series
            if isinstance(time_UTC, (list, pd.Series)):
                is_batch_query = True
            # Check if lat/lon are lists/Series
            elif lat is not None and lon is not None:
                if isinstance(lat, (list, pd.Series)) or isinstance(lon, (list, pd.Series)):
                    is_batch_query = True
            # Check if geometry is a GeoSeries
            elif geometry is not None and isinstance(geometry, gpd.GeoSeries):
                is_batch_query = True
        
        # Handle vectorized batch queries
        if is_batch_query:
            if not HAS_OPENDAP_SUPPORT:
                raise ImportError(
                    "Point query support requires xarray and netCDF4. "
                    "Install with: conda install -c conda-forge xarray netcdf4"
                )
            
            # Convert inputs to lists
            if isinstance(time_UTC, pd.Series):
                times = time_UTC.tolist()
            elif isinstance(time_UTC, list):
                times = time_UTC
            else:
                raise ValueError("For batch queries, time_UTC must be a list or Series")
            
            # Get geometries
            if geometry is not None:
                if isinstance(geometry, gpd.GeoSeries):
                    geometries = geometry.tolist()
                elif isinstance(geometry, list):
                    geometries = geometry
                else:
                    raise ValueError("For batch queries with geometry, must provide GeoSeries or list")
            elif lat is not None and lon is not None:
                # Convert lat/lon to Point geometries
                if isinstance(lat, pd.Series):
                    lats = lat.tolist()
                else:
                    lats = lat if isinstance(lat, list) else [lat]
                
                if isinstance(lon, pd.Series):
                    lons = lon.tolist()
                else:
                    lons = lon if isinstance(lon, list) else [lon]
                
                if len(lats) != len(lons):
                    raise ValueError("lat and lon must have the same length")
                
                geometries = [Point(lon_val, lat_val) for lon_val, lat_val in zip(lons, lats)]
            else:
                raise ValueError("For batch queries, must provide geometry or lat/lon")
            
            # Validate lengths match
            if len(times) != len(geometries):
                raise ValueError(
                    f"Number of times ({len(times)}) must match number of geometries ({len(geometries)})"
                )
            
            # Parse time strings to datetime objects
            parsed_times = []
            for t in times:
                if isinstance(t, str):
                    parsed_times.append(parser.parse(t))
                else:
                    parsed_times.append(t)
            
            # Optimize queries: group by unique coordinates, then cluster times
            # to avoid querying excessively long time ranges
            
            # Build mapping of (lat, lon) -> list of (index, time, geometry)
            coord_to_records = {}
            for idx, (time_val, geom) in enumerate(zip(parsed_times, geometries)):
                # Extract point coordinates
                if isinstance(geom, Point):
                    pt_lon, pt_lat = geom.x, geom.y
                elif isinstance(geom, MultiPoint):
                    # Use first point
                    pt_lon, pt_lat = geom.geoms[0].x, geom.geoms[0].y
                else:
                    raise ValueError(f"Unsupported geometry type: {type(geom)}")
                
                # Round to avoid floating point issues
                coord_key = (round(pt_lat, 6), round(pt_lon, 6))
                
                if coord_key not in coord_to_records:
                    coord_to_records[coord_key] = []
                
                coord_to_records[coord_key].append({
                    'index': idx,
                    'time': time_val,
                    'geometry': geom
                })
            
            # Function to cluster times at a coordinate to avoid long queries
            def cluster_times(records, max_days_per_query=30):
                """
                Cluster records by time to keep queries under max_days_per_query duration.
                Returns list of record clusters.
                """
                if not records:
                    return []
                
                # Sort by time
                sorted_records = sorted(records, key=lambda r: r['time'])
                
                clusters = []
                current_cluster = [sorted_records[0]]
                
                for record in sorted_records[1:]:
                    cluster_start = current_cluster[0]['time']
                    cluster_end = current_cluster[-1]['time']
                    record_time = record['time']
                    
                    # Check if adding this record would exceed max duration
                    potential_span = (max(cluster_end, record_time) - 
                                    min(cluster_start, record_time))
                    
                    if potential_span.total_seconds() / 86400 <= max_days_per_query:
                        # Add to current cluster
                        current_cluster.append(record)
                    else:
                        # Start new cluster
                        clusters.append(current_cluster)
                        current_cluster = [record]
                
                # Add final cluster
                if current_cluster:
                    clusters.append(current_cluster)
                
                return clusters
            
            # Count total queries needed
            total_query_batches = 0
            for coord_key, records in coord_to_records.items():
                clusters = cluster_times(records, max_days_per_query=30)
                total_query_batches += len(clusters)
            
            logger.info(
                f"Processing {len(parsed_times)} spatio-temporal records at "
                f"{len(coord_to_records)} unique coordinates "
                f"({total_query_batches} query batches)..."
            )
            
            # Initialize results dictionary indexed by original record index
            results_by_index = {i: {'time_UTC': t, 'geometry': g} 
                               for i, (t, g) in enumerate(zip(parsed_times, geometries))}
            
            # Process each variable
            for var_name in variable_names:
                # Determine dataset for this variable
                var_dataset = dataset
                if var_dataset is None:
                    if var_name in GEOS5FP_VARIABLES:
                        _, var_dataset, raw_variable = self._get_variable_info(var_name)
                    else:
                        raise ValueError(
                            f"Dataset must be specified when using raw variable name '{var_name}'. "
                            f"Known variables: {list(GEOS5FP_VARIABLES.keys())}"
                        )
                else:
                    # Use provided dataset, determine variable
                    if var_name in GEOS5FP_VARIABLES:
                        _, _, raw_variable = self._get_variable_info(var_name)
                    else:
                        raw_variable = var_name
                
                variable_opendap = raw_variable.lower()
                
                logger.info(
                    f"Querying {var_name} from {var_dataset} "
                    f"at {len(coord_to_records)} coordinates ({total_query_batches} batches)..."
                )
                
                batch_num = 0
                
                # Query each unique coordinate with time clustering
                for coord_idx, (coord_key, records) in enumerate(coord_to_records.items(), 1):
                    pt_lat, pt_lon = coord_key
                    
                    # Cluster times to keep queries manageable
                    time_clusters = cluster_times(records, max_days_per_query=30)
                    
                    logger.info(
                        f"  Coordinate {coord_idx}/{len(coord_to_records)}: "
                        f"({pt_lat:.4f}, {pt_lon:.4f}) - "
                        f"{len(records)} records in {len(time_clusters)} time clusters"
                    )
                    
                    # Query each time cluster
                    for cluster_idx, cluster in enumerate(time_clusters, 1):
                        batch_num += 1
                        
                        # Get time range for this cluster
                        times_in_cluster = [r['time'] for r in cluster]
                        min_time = min(times_in_cluster)
                        max_time = max(times_in_cluster)
                        time_span_days = (max_time - min_time).total_seconds() / 86400
                        
                        # Add buffer to ensure we get data
                        time_range_start = min_time - timedelta(hours=2)
                        time_range_end = max_time + timedelta(hours=2)
                        
                        logger.info(
                            f"    Batch {batch_num}/{total_query_batches}: "
                            f"cluster {cluster_idx}/{len(time_clusters)} - "
                            f"{len(cluster)} records, {time_span_days:.1f} days "
                            f"({min_time.date()} to {max_time.date()})"
                        )
                        
                        try:
                            # Query for this time cluster at this coordinate
                            result = query_geos5fp_point(
                                dataset=var_dataset,
                                variable=variable_opendap,
                                lat=pt_lat,
                                lon=pt_lon,
                                time_range=(time_range_start, time_range_end),
                                dropna=dropna
                            )
                            
                            if len(result.df) == 0:
                                logger.warning(
                                    f"No data found for ({pt_lat}, {pt_lon}) "
                                    f"in time range {time_range_start.date()} to {time_range_end.date()}"
                                )
                                # Set all records in this cluster to None
                                for record in cluster:
                                    results_by_index[record['index']][var_name] = None
                            else:
                                # Extract values for each needed time in this cluster
                                for record in cluster:
                                    target_time = record['time']
                                    
                                    # Find closest available time
                                    time_diffs = abs(result.df.index - target_time)
                                    closest_idx = time_diffs.argmin()
                                    value = result.df.iloc[closest_idx][variable_opendap]
                                    
                                    # Store in results
                                    results_by_index[record['index']][var_name] = value
                        
                        except Exception as e:
                            logger.warning(
                                f"Failed to query {var_name} at ({pt_lat}, {pt_lon}): {e}"
                            )
                            # Set all records in this cluster to None
                            for record in cluster:
                                results_by_index[record['index']][var_name] = None
            
            # Convert results dictionary to list in original order
            all_dfs = [results_by_index[i] for i in range(len(parsed_times))]
            
            # Create DataFrame from results
            result_df = pd.DataFrame(all_dfs)
            
            # Set time as index
            result_df = result_df.set_index('time_UTC')
            
            # Move geometry to end
            if 'geometry' in result_df.columns:
                cols = [c for c in result_df.columns if c != 'geometry']
                cols.append('geometry')
                result_df = result_df[cols]
            
            # Convert to GeoDataFrame
            result_gdf = gpd.GeoDataFrame(result_df, geometry='geometry', crs='EPSG:4326')
            
            return result_gdf
        
        # Create Point geometry from lat/lon if provided
        if lat is not None and lon is not None:
            if geometry is not None:
                raise ValueError("Cannot specify both geometry and lat/lon")
            geometry = Point(lon, lat)
        
        # Determine if this is a point query with time range
        is_point_time_series = (
            time_range is not None and 
            (self._is_point_geometry(geometry) or (lat is not None and lon is not None))
        )
        
        # Check if multiple variables requested for non-point query
        if not single_variable and not self._is_point_geometry(geometry):
            raise ValueError("Multiple variables can only be queried for point geometries")
        
        # Handle point time-series queries with OPeNDAP
        if is_point_time_series:
            if not HAS_OPENDAP_SUPPORT:
                raise ImportError(
                    "Point query support requires xarray and netCDF4. "
                    "Install with: conda install -c conda-forge xarray netcdf4"
                )
            
            # Extract point coordinates
            if geometry is not None:
                points = self._extract_points(geometry)
            else:
                points = [(lat, lon)]
            
            # Process each variable
            all_variable_dfs = []
            
            for var_name in variable_names:
                # Determine dataset for this variable
                var_dataset = dataset
                if var_dataset is None:
                    if var_name in GEOS5FP_VARIABLES:
                        _, var_dataset, raw_variable = self._get_variable_info(var_name)
                    else:
                        raise ValueError(
                            f"Dataset must be specified when using raw variable name '{var_name}'. "
                            f"Known variables: {list(GEOS5FP_VARIABLES.keys())}"
                        )
                else:
                    # Use provided dataset, determine variable
                    if var_name in GEOS5FP_VARIABLES:
                        _, _, raw_variable = self._get_variable_info(var_name)
                    else:
                        raw_variable = var_name
                
                # Convert variable name to lowercase for OPeNDAP
                variable_opendap = raw_variable.lower()
                
                logger.info(
                    f"retrieving {var_name} time-series "
                    f"from GEOS-5 FP {var_dataset} {raw_variable} "
                    f"for time range {time_range[0]} to {time_range[1]}"
                )
                
                # Query each point for this variable
                dfs = []
                for pt_lat, pt_lon in points:
                    try:
                        result = query_geos5fp_point(
                            dataset=var_dataset,
                            variable=variable_opendap,
                            lat=pt_lat,
                            lon=pt_lon,
                            time_range=time_range,
                            dropna=dropna
                        )
                        
                        if len(result.df) == 0:
                            logger.warning(f"No data found for point ({pt_lat}, {pt_lon})")
                            continue
                        
                        df_point = result.df.copy()
                        
                        # Rename from OPeNDAP variable name to requested variable name
                        if variable_opendap in df_point.columns:
                            df_point = df_point.rename(columns={variable_opendap: var_name})
                        
                        # Add geometry column at the end
                        df_point['geometry'] = Point(pt_lon, pt_lat)
                        
                        dfs.append(df_point)
                    except Exception as e:
                        logger.warning(f"Failed to query point ({pt_lat}, {pt_lon}) for {var_name}: {e}")
                
                if not dfs:
                    logger.warning(f"No successful point queries for variable {var_name}")
                    continue
                
                var_df = pd.concat(dfs, ignore_index=False)
                all_variable_dfs.append(var_df)
            
            if not all_variable_dfs:
                raise ValueError("No successful point queries for any variable")
            
            # Merge all variable DataFrames
            if len(all_variable_dfs) == 1:
                result_df = all_variable_dfs[0]
            else:
                # Save geometry from first dataframe
                geometry_col = all_variable_dfs[0]['geometry'].copy()
                
                # Merge on index (time), excluding geometry from all but first
                result_df = all_variable_dfs[0].drop(columns=['geometry'])
                for var_df in all_variable_dfs[1:]:
                    # Get the variable column name (exclude geometry)
                    var_cols = [col for col in var_df.columns if col != 'geometry']
                    
                    # Merge on index
                    result_df = result_df.merge(
                        var_df[var_cols],
                        left_index=True,
                        right_index=True,
                        how='outer'
                    )
                
                # Add geometry column at the end
                result_df['geometry'] = geometry_col
            
            # Convert to GeoDataFrame
            result_gdf = gpd.GeoDataFrame(result_df, geometry='geometry', crs='EPSG:4326')
            
            return result_gdf
        
        # Handle single-time queries (raster or single-point)
        else:
            if time_UTC is None:
                raise ValueError("time_UTC is required for single-time queries")
            
            # For single variable, use original logic
            if single_variable:
                var_name = variable_names[0]
                
                # Check if variable is in our predefined list
                if var_name in GEOS5FP_VARIABLES:
                    # Use the standard retrieval method
                    return self._get_simple_variable(
                        var_name,
                        time_UTC,
                        geometry=geometry,
                        resampling=resampling,
                        **kwargs
                    )
                
                # Raw variable name provided
                else:
                    if dataset is None:
                        raise ValueError(
                            f"Dataset must be specified when using raw variable name '{var_name}'. "
                            f"Known variables: {list(GEOS5FP_VARIABLES.keys())}"
                        )
                    
                    if isinstance(time_UTC, str):
                        time_UTC = parser.parse(time_UTC)
                    
                    # Check if this is a point query
                    if self._is_point_geometry(geometry):
                        if not HAS_OPENDAP_SUPPORT:
                            raise ImportError(
                                "Point query support requires xarray and netCDF4. "
                                "Install with: conda install -c conda-forge xarray netcdf4"
                            )
                        
                        logger.info(
                            f"retrieving {var_name} "
                            f"from GEOS-5 FP {dataset} at point location(s)"
                        )
                        
                        points = self._extract_points(geometry)
                        dfs = []
                        variable_opendap = var_name.lower()
                        
                        for pt_lat, pt_lon in points:
                            try:
                                # Query small time window around target time
                                time_window_start = time_UTC - timedelta(hours=2)
                                time_window_end = time_UTC + timedelta(hours=2)
                                
                                result = query_geos5fp_point(
                                    dataset=dataset,
                                    variable=variable_opendap,
                                    lat=pt_lat,
                                    lon=pt_lon,
                                    time_range=(time_window_start, time_window_end),
                                    dropna=dropna
                                )
                                
                                if len(result.df) == 0:
                                    logger.warning(f"No data found for point ({pt_lat}, {pt_lon})")
                                    continue
                                
                                # Find closest time
                                time_diffs = abs(result.df.index - time_UTC)
                                closest_idx = time_diffs.argmin()
                                df_point = result.df.iloc[[closest_idx]].copy()
                                
                                if variable_opendap in df_point.columns:
                                    df_point = df_point.rename(columns={variable_opendap: var_name})
                                
                                # Add geometry column at the end
                                df_point['geometry'] = Point(pt_lon, pt_lat)
                                
                                dfs.append(df_point)
                            except Exception as e:
                                logger.warning(f"Failed to query point ({pt_lat}, {pt_lon}): {e}")
                        
                        if not dfs:
                            raise ValueError("No successful point queries")
                        
                        result_df = pd.concat(dfs, ignore_index=False)
                        result_gdf = gpd.GeoDataFrame(result_df, geometry='geometry', crs='EPSG:4326')
                        return result_gdf
                    
                    # Raster query
                    else:
                        logger.info(
                            f"retrieving {var_name} "
                            f"from GEOS-5 FP {dataset} " +
                            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
                        )
                        
                        return self.interpolate(
                            time_UTC=time_UTC,
                            product=dataset,
                            variable=var_name,
                            geometry=geometry,
                            resampling=resampling,
                            **kwargs
                        )
            
            # Multiple variables at single time (point query only)
            else:
                if not self._is_point_geometry(geometry):
                    raise ValueError("Multiple variables can only be queried for point geometries")
                
                if not HAS_OPENDAP_SUPPORT:
                    raise ImportError(
                        "Point query support requires xarray and netCDF4. "
                        "Install with: conda install -c conda-forge xarray netcdf4"
                    )
                
                if isinstance(time_UTC, str):
                    time_UTC = parser.parse(time_UTC)
                
                points = self._extract_points(geometry)
                all_variable_dfs = []
                
                # Process each variable
                for var_name in variable_names:
                    # Determine dataset for this variable
                    var_dataset = dataset
                    if var_dataset is None:
                        if var_name in GEOS5FP_VARIABLES:
                            _, var_dataset, raw_variable = self._get_variable_info(var_name)
                        else:
                            raise ValueError(
                                f"Dataset must be specified when using raw variable name '{var_name}'. "
                                f"Known variables: {list(GEOS5FP_VARIABLES.keys())}"
                            )
                    else:
                        # Use provided dataset, determine variable
                        if var_name in GEOS5FP_VARIABLES:
                            _, _, raw_variable = self._get_variable_info(var_name)
                        else:
                            raw_variable = var_name
                    
                    variable_opendap = raw_variable.lower()
                    
                    logger.info(
                        f"retrieving {var_name} "
                        f"from GEOS-5 FP {var_dataset} at point location(s)"
                    )
                    
                    dfs = []
                    for pt_lat, pt_lon in points:
                        try:
                            # Query small time window around target time
                            time_window_start = time_UTC - timedelta(hours=2)
                            time_window_end = time_UTC + timedelta(hours=2)
                            
                            result = query_geos5fp_point(
                                dataset=var_dataset,
                                variable=variable_opendap,
                                lat=pt_lat,
                                lon=pt_lon,
                                time_range=(time_window_start, time_window_end),
                                dropna=dropna
                            )
                            
                            if len(result.df) == 0:
                                logger.warning(f"No data found for point ({pt_lat}, {pt_lon})")
                                continue
                            
                            # Find closest time
                            time_diffs = abs(result.df.index - time_UTC)
                            closest_idx = time_diffs.argmin()
                            df_point = result.df.iloc[[closest_idx]].copy()
                            
                            if variable_opendap in df_point.columns:
                                df_point = df_point.rename(columns={variable_opendap: var_name})
                            
                            # Add geometry column at the end
                            df_point['geometry'] = Point(pt_lon, pt_lat)
                            
                            dfs.append(df_point)
                        except Exception as e:
                            logger.warning(f"Failed to query point ({pt_lat}, {pt_lon}) for {var_name}: {e}")
                    
                    if not dfs:
                        logger.warning(f"No successful point queries for variable {var_name}")
                        continue
                    
                    var_df = pd.concat(dfs, ignore_index=False)
                    all_variable_dfs.append(var_df)
                
                if not all_variable_dfs:
                    raise ValueError("No successful point queries for any variable")
                
                # Merge all variable DataFrames
                if len(all_variable_dfs) == 1:
                    result_df = all_variable_dfs[0]
                else:
                    # Save geometry from first dataframe
                    geometry_col = all_variable_dfs[0]['geometry'].copy()
                    
                    # Merge on index (time), excluding geometry from all but first
                    result_df = all_variable_dfs[0].drop(columns=['geometry'])
                    for var_df in all_variable_dfs[1:]:
                        # Get the variable column name (exclude geometry)
                        var_cols = [col for col in var_df.columns if col != 'geometry']
                        
                        # Merge on index
                        result_df = result_df.merge(
                            var_df[var_cols],
                            left_index=True,
                            right_index=True,
                            how='outer'
                        )
                    
                    # Add geometry column at the end
                    result_df['geometry'] = geometry_col
                
                # Convert to GeoDataFrame
                result_gdf = gpd.GeoDataFrame(result_df, geometry='geometry', crs='EPSG:4326')
                
                return result_gdf
