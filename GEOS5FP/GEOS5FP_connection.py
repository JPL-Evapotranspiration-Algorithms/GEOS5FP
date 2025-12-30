import json
import logging
import os
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
from requests.exceptions import ChunkedEncodingError, ConnectionError, SSLError
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
from .create_robust_session import create_robust_session
from .create_legacy_ssl_context import create_legacy_ssl_context
from .make_head_request_with_ssl_fallback import make_head_request_with_ssl_fallback
from .download_file import download_file
from .get_variable_info import get_variable_info
from .geometry_utils import is_point_geometry, extract_points

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
            save_products: bool = False,
            verbose: bool = True,
            use_opendap: bool = True,
            allow_fallback: bool = False):
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
        self.verbose = verbose
        self.use_opendap = use_opendap and HAS_OPENDAP_SUPPORT
        self.allow_fallback = allow_fallback

    def __repr__(self):
        display_dict = {
            "URL": self.remote,
            "download_directory": self.download_directory
        }

        display_string = json.dumps(display_dict, indent=2)

        return display_string

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

    def _get_variable_info(self, variable_name: str) -> Tuple[str, str, str]:
        """
        Get information about a GEOS-5 FP variable.
        
        Args:
            variable_name: Name or alias of the variable
            
        Returns:
            Tuple of (description, product, variable)
            
        Raises:
            ValueError: If variable name is invalid
        """
        return get_variable_info(variable_name)

    def _is_point_geometry(self, geometry: Any) -> bool:
        """
        Check if a geometry is a point or multipoint.
        
        Args:
            geometry: Shapely geometry object or other input
            
        Returns:
            True if geometry is Point or MultiPoint, False otherwise
        """
        return is_point_geometry(geometry)

    def _extract_points(self, geometry: Any) -> List[Tuple[float, float]]:
        """
        Extract (lon, lat) coordinates from point geometry.
        
        Args:
            geometry: Point or MultiPoint geometry
            
        Returns:
            List of (longitude, latitude) tuples
            
        Raises:
            ValueError: If geometry is not a point type
        """
        return extract_points(geometry)

    @property
    def years_available(self) -> List[date]:
        listing = self.list_remote_directory(self.remote)
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

        listing = self.list_remote_directory(year_URL)
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

        listing = self.list_remote_directory(month_URL)
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
                return self.list_GEOS5FP_granules(self.latest_date_available).sort_values(by="time_UTC").iloc[-1].time_UTC.to_pydatetime()
            except Exception as e:
                logger.warning(e)
                sleep(wait_seconds)
                continue


    def time_from_URL(self, URL: str) -> datetime:
        return datetime.strptime(URL.split(".")[-3], "%Y%m%d_%H%M")

    def list_remote_directory(self, URL: str, timeout: float = None, retries: int = None) -> List[str]:
        """Fetch and cache the contents of a remote directory."""
        if URL in self._listings:
            return self._listings[URL]
        else:
            listing = HTTP_listing(URL, timeout=timeout, retries=retries)
            self._listings[URL] = listing

            return listing

    def list_GEOS5FP_granules(
            self,
            date_UTC: Union[datetime, str],
            product_name: str = None,
            timeout: float = None,
            retries: int = None) -> pd.DataFrame:
        """Get GEOS-5 FP granule metadata for a specific day."""
        if timeout is None:
            timeout = self.DEFAULT_TIMEOUT_SECONDS

        if retries is None:
            retries = self.DEFAULT_RETRIES

        day_URL = self.day_URL(date_UTC)

        if requests.head(day_URL).status_code == 404:
            raise GEOS5FPDayNotAvailable(f"GEOS-5 FP day not available: {day_URL}")

        logger.info(f"listing URL: {cl.URL(day_URL)}")
        # listing = HTTP_listing(day_URL, timeout=timeout, retries=retries)
        listing = self.list_remote_directory(day_URL, timeout=timeout, retries=retries)

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
            return self.list_GEOS5FP_granules(
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

    def download_file(self, URL: str, filename: str = None, retries: int = 3, wait_seconds: int = 30) -> 'GEOS5FPGranule':
        """
        Download a GEOS-5 FP file with GEOS-5 FP specific handling and comprehensive validation.
        
        This method includes:
        - Pre-download validation of existing files
        - Automatic cleanup of invalid existing files
        - Post-download validation with retry logic
        - Comprehensive error reporting
        
        Communicates specific circumstances using exception classes.
        """
        from .exceptions import GEOS5FPDayNotAvailable, GEOS5FPGranuleNotAvailable, FailedGEOS5FPDownload, GEOS5FPSSLError
        from .validate_GEOS5FP_NetCDF_file import validate_GEOS5FP_NetCDF_file



        # GEOS-5 FP: check remote existence and generate filename if needed
        try:
            head_resp = make_head_request_with_ssl_fallback(URL)
        except SSLError as e:
            logger.error(f"SSL connection failed for URL: {URL}")
            logger.error(f"SSL error details: {e}")
            raise GEOS5FPSSLError(f"Failed to establish SSL connection", original_error=e, url=URL)
        except Exception as e:
            logger.error(f"Failed to check remote file existence: {e}")
            raise FailedGEOS5FPDownload(f"Connection error: {e}")
        if head_resp.status_code == 404:
            directory_URL = posixpath.dirname(URL)
            try:
                dir_resp = make_head_request_with_ssl_fallback(directory_URL)
                if dir_resp.status_code == 404:
                    raise GEOS5FPDayNotAvailable(directory_URL)
                else:
                    raise GEOS5FPGranuleNotAvailable(URL)
            except SSLError as e:
                logger.error(f"SSL connection failed for directory URL: {directory_URL}")
                raise GEOS5FPSSLError(f"Failed to establish SSL connection to directory", original_error=e, url=directory_URL)
            except Exception as e:
                logger.error(f"Failed to check directory existence: {e}")
                raise FailedGEOS5FPDownload(f"Connection error checking directory: {e}")

        if filename is None:
            filename = self.download_filename(URL)

        expanded_filename = os.path.expanduser(filename)

        # Pre-download validation: check if file already exists and is valid
        if exists(expanded_filename):
            logger.info(f"checking existing file: {filename}")
            validation_result = validate_GEOS5FP_NetCDF_file(expanded_filename, verbose=False)
            
            if validation_result.is_valid:
                logger.info(f"existing file is valid: {filename} ({validation_result.metadata.get('file_size_mb', 'unknown')} MB)")
                return GEOS5FPGranule(filename)
            else:
                logger.warning(f"existing file is invalid, removing: {filename}")
                for error in validation_result.errors[:3]:  # Log first 3 errors
                    logger.warning(f"  validation error: {error}")
                try:
                    os.remove(expanded_filename)
                    logger.info(f"removed invalid file: {filename}")
                except OSError as e:
                    logger.warning(f"failed to remove invalid file {filename}: {e}")

        # Track download attempts with validation
        download_attempts = 0
        max_download_attempts = retries
        
        while download_attempts < max_download_attempts:
            download_attempts += 1
            logger.info(f"download attempt {download_attempts}/{max_download_attempts}: {URL}")
            
            try:
                result_filename = download_file(
                    URL=URL,
                    filename=filename,
                    retries=1,  # Handle retries at this level
                    wait_seconds=wait_seconds
                )
            except FailedGEOS5FPDownload as e:
                # Already a specific download failure
                if download_attempts >= max_download_attempts:
                    raise
                logger.warning(f"download attempt {download_attempts} failed: {e}")
                logger.warning(f"waiting {wait_seconds} seconds before retry...")
                sleep(wait_seconds)
                continue
            except Exception as e:
                # Any other error during download
                if download_attempts >= max_download_attempts:
                    raise FailedGEOS5FPDownload(str(e))
                logger.warning(f"download attempt {download_attempts} failed: {e}")
                logger.warning(f"waiting {wait_seconds} seconds before retry...")
                sleep(wait_seconds)
                continue

            # Post-download validation with comprehensive checks
            logger.info(f"validating downloaded file: {result_filename}")
            validation_result = validate_GEOS5FP_NetCDF_file(expanded_filename, verbose=False)
            
            if validation_result.is_valid:
                logger.info(f"download and validation successful: {result_filename} ({validation_result.metadata.get('file_size_mb', 'unknown')} MB)")
                if 'product_name' in validation_result.metadata:
                    logger.info(f"validated product: {validation_result.metadata['product_name']}")
                return GEOS5FPGranule(result_filename)
            else:
                logger.warning(f"downloaded file failed validation: {result_filename}")
                for error in validation_result.errors[:3]:  # Log first 3 errors
                    logger.warning(f"  validation error: {error}")
                
                # Clean up invalid download
                try:
                    os.remove(expanded_filename)
                    logger.info(f"removed invalid downloaded file: {result_filename}")
                except OSError as e:
                    logger.warning(f"failed to remove invalid file {result_filename}: {e}")
                
                if download_attempts >= max_download_attempts:
                    error_summary = '; '.join(validation_result.errors[:2])
                    raise FailedGEOS5FPDownload(f"downloaded file validation failed after {max_download_attempts} attempts: {error_summary}")
                
                logger.warning(f"retrying download due to validation failure (attempt {download_attempts + 1}/{max_download_attempts})")
                logger.warning(f"waiting {wait_seconds} seconds before retry...")
                sleep(wait_seconds)

        # This should not be reached due to the logic above, but included for completeness
        raise FailedGEOS5FPDownload(f"download failed after {max_download_attempts} attempts")

    def before_and_after(
            self,
            time_UTC: Union[datetime, str],
            product: str,
            interval: int = None,
            expected_hours: List[float] = None,
            timeout: float = None,
            retries: int = None,
            use_http_listing: bool = DEFAULT_USE_HTTP_LISTING) -> Tuple[GEOS5FPGranule, GEOS5FPGranule]:
        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)

        ## FIXME need to check local filesystem for existing files here first before searching remote

        search_date = time_UTC.date()
        logger.info(f"searching GEOS-5 FP {cl.name(product)} at " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M:%S} UTC"))

        logger.info(f"listing available {product} files for {search_date}...")
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

        logger.info(f"found {len(product_listing)} {product} files, finding bracketing times...")
        before_listing = product_listing[product_listing.time_UTC < time_UTC]

        if len(before_listing) == 0:
            raise IOError(f"no {product} files found preceeding {time_UTC}")

        before_time_UTC, before_URL = before_listing.iloc[-1][["time_UTC", "URL"]]
        logger.info(f"before time: {cl.time(f'{before_time_UTC:%Y-%m-%d %H:%M} UTC')}")
        
        after_listing = product_listing[product_listing.time_UTC > time_UTC]

        if len(after_listing) == 0:
            logger.info(f"no after files in current day, checking next day...")
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
        logger.info(f"after time: {cl.time(f'{after_time_UTC:%Y-%m-%d %H:%M} UTC')}")
        
        logger.info(f"downloading/loading before granule...")
        before_granule = self.download_file(before_URL)
        
        logger.info(f"downloading/loading after granule...")
        after_granule = self.download_file(after_URL)

        return before_granule, after_granule

    def variable(
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
        Retrieve a single predefined variable at a specific time.
        
        This is a convenience method for retrieving individual variables from the
        GEOS5FP_VARIABLES registry. For more advanced queries including multiple
        variables, time ranges, or batch queries, use the query() method instead.
        
        Parameters
        ----------
        variable_name : str
            Name of the variable (must exist in GEOS5FP_VARIABLES)
        time_UTC : datetime or str
            Date/time in UTC
        geometry : RasterGeometry or Point or MultiPoint, optional
            Target geometry (can be raster geometry or Point/MultiPoint for time-series)
        resampling : str, optional
            Sampling method for resampling to target geometry
        interval : int, optional
            Interval for product listing
        expected_hours : list of float, optional
            Expected hours for product listing
        min_value : Any, optional
            Minimum value for interpolation
        max_value : Any, optional
            Maximum value for interpolation
        exclude_values : optional
            Values to exclude in interpolation
        cmap : optional
            Colormap for the result
        clip_min : Any, optional
            Minimum value for final clipping (applied after interpolation)
        clip_max : Any, optional
            Maximum value for final clipping (applied after interpolation)
        
        Returns
        -------
        Raster or DataFrame
            Raster of the requested variable or DataFrame for point queries
        
        See Also
        --------
        query : More flexible query method supporting multiple variables and time ranges
        """
        # Handle pandas Series, DatetimeIndex, or array-like time_UTC inputs
        if isinstance(time_UTC, (pd.Series, pd.DatetimeIndex, list, tuple, np.ndarray)):
            from time import time as current_time
            
            logger.info(f"Processing {len(time_UTC)} time values for variable {variable_name}")
            
            # Convert to pandas Series if not already
            if not isinstance(time_UTC, pd.Series):
                time_series = pd.Series(time_UTC)
            else:
                time_series = time_UTC
            
            # Handle geometry - can be array-like or single geometry
            if hasattr(geometry, '__len__') and not isinstance(geometry, (str, RasterGeometry)):
                # geometry is array-like (e.g., GeometryArray, list of Points)
                if len(time_series) != len(geometry):
                    raise ValueError(
                        f"Length mismatch: time_UTC has {len(time_series)} values "
                        f"but geometry has {len(geometry)} elements"
                    )
                
                # Process each (time, geometry) pair with progress tracking
                results = []
                start_time = current_time()
                total_pairs = len(time_series)
                
                for idx, (time_val, geom_val) in enumerate(zip(time_series, geometry)):
                    # Calculate ETA
                    if idx > 0:
                        elapsed = current_time() - start_time
                        avg_time = elapsed / idx
                        remaining = total_pairs - idx
                        eta_seconds = avg_time * remaining
                        
                        if eta_seconds < 60:
                            eta_str = f"{eta_seconds:.0f}s"
                        elif eta_seconds < 3600:
                            eta_str = f"{eta_seconds/60:.1f}m"
                        else:
                            eta_str = f"{eta_seconds/3600:.1f}h"
                        
                        logger.info(f"Processing pair {idx+1}/{total_pairs} [{100*idx/total_pairs:.1f}%]: time={time_val}, geometry={geom_val} (ETA: {eta_str})")
                    else:
                        logger.info(f"Processing pair {idx+1}/{total_pairs}: time={time_val}, geometry={geom_val}")
                    
                    result = self.variable(
                        variable_name=variable_name,
                        time_UTC=time_val,
                        geometry=geom_val,
                        resampling=resampling,
                        interval=interval,
                        expected_hours=expected_hours,
                        min_value=min_value,
                        max_value=max_value,
                        exclude_values=exclude_values,
                        cmap=cmap,
                        clip_min=clip_min,
                        clip_max=clip_max
                    )
                    results.append(result)
                
                logger.info(f"Completed processing {total_pairs} pairs in {current_time() - start_time:.1f}s")
                
                # Combine results
                if all(isinstance(r, pd.DataFrame) for r in results):
                    # Concatenate DataFrames
                    combined = pd.concat(results, ignore_index=True)
                    return combined
                elif all(isinstance(r, (Raster, np.ndarray)) for r in results):
                    # Stack rasters
                    return np.array(results)
                else:
                    # Mixed types - return as list
                    return results
            else:
                # Single geometry for all times - process each unique time with progress tracking
                unique_times = time_series.unique()
                time_to_result = {}
                start_time = current_time()
                total_unique = len(unique_times)
                
                logger.info(f"Processing {total_unique} unique timestamps out of {len(time_series)} total values")
                
                for idx, unique_time in enumerate(unique_times):
                    # Calculate ETA
                    if idx > 0:
                        elapsed = current_time() - start_time
                        avg_time = elapsed / idx
                        remaining = total_unique - idx
                        eta_seconds = avg_time * remaining
                        
                        if eta_seconds < 60:
                            eta_str = f"{eta_seconds:.0f}s"
                        elif eta_seconds < 3600:
                            eta_str = f"{eta_seconds/60:.1f}m"
                        else:
                            eta_str = f"{eta_seconds/3600:.1f}h"
                        
                        logger.info(f"Processing timestamp {idx+1}/{total_unique} [{100*idx/total_unique:.1f}%]: {unique_time} (ETA: {eta_str})")
                    else:
                        logger.info(f"Processing timestamp {idx+1}/{total_unique}: {unique_time}")
                    
                    result = self.variable(
                        variable_name=variable_name,
                        time_UTC=unique_time,
                        geometry=geometry,
                        resampling=resampling,
                        interval=interval,
                        expected_hours=expected_hours,
                        min_value=min_value,
                        max_value=max_value,
                        exclude_values=exclude_values,
                        cmap=cmap,
                        clip_min=clip_min,
                        clip_max=clip_max
                    )
                    time_to_result[unique_time] = result
                
                logger.info(f"Completed processing {total_unique} unique timestamps in {current_time() - start_time:.1f}s")
                
                # Map results back to original time series
                logger.info(f"Mapping results to original {len(time_series)} time series entries")
                results = [time_to_result[t] for t in time_series]
                
                # Combine results
                if all(isinstance(r, pd.DataFrame) for r in results):
                    combined = pd.concat(results, ignore_index=True)
                    return combined
                elif all(isinstance(r, (Raster, np.ndarray)) for r in results):
                    return np.array(results)
                else:
                    return results
        
        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)
            
        NAME, PRODUCT, VARIABLE = get_variable_info(variable_name)
        
        # Check if this is a point query
        if is_point_geometry(geometry):
            # Skip OPeNDAP if disabled or not available
            if not self.use_opendap:
                logger.info(f"OPeNDAP disabled, using standard interpolation for point query")
            elif not HAS_OPENDAP_SUPPORT:
                logger.warning(
                    "Point query requires xarray and netCDF4. "
                    "Install with: conda install -c conda-forge xarray netcdf4. "
                    "Falling back to standard interpolation."
                )
            else:
                # Try OPeNDAP point query
                try:
                    return self._query_point_via_opendap(
                        NAME, PRODUCT, VARIABLE, variable_name, 
                        time_UTC, geometry, resampling, interval, 
                        expected_hours, min_value, max_value, 
                        exclude_values, cmap, clip_min, clip_max
                    )
                except Exception as e:
                    if self.allow_fallback:
                        logger.warning(f"OPeNDAP point query failed: {e}")
                        logger.info("Falling back to standard interpolation method")
                    else:
                        logger.error(f"OPeNDAP point query failed: {e}")
                        logger.error("Fallback to file download is disabled. Set allow_fallback=True to enable.")
                        raise
            
            # Fallback to standard interpolation for point queries (only if allow_fallback=True)
            logger.info(
                f"retrieving {cl.name(NAME)} "
                f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
                "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC") +
                " using interpolation method"
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

    def _query_point_via_opendap(
            self,
            NAME: str,
            PRODUCT: str,
            VARIABLE: str,
            variable_name: str,
            time_UTC: datetime,
            geometry: Any,
            resampling: str,
            interval: int,
            expected_hours: List[float],
            min_value: Any,
            max_value: Any,
            exclude_values: Any,
            cmap: Any,
            clip_min: Any,
            clip_max: Any) -> pd.DataFrame:
        """Query point(s) via OPeNDAP with fallback to interpolation."""
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
        
        points = extract_points(geometry)
        dfs = []
        
        # OPeNDAP uses lowercase variable names
        variable_opendap = VARIABLE.lower()
        
        # Track failed points for fallback
        failed_points = []
        
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
                    dropna=True,
                    retries=5,
                    retry_delay=15.0
                )
                
                if len(result.df) == 0:
                    logger.warning(f"No data found for point ({lat}, {lon}) near {time_UTC}")
                    failed_points.append((lat, lon))
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
                logger.warning(f"Failed to query point ({lat}, {lon}) via OPeNDAP: {e}")
                failed_points.append((lat, lon))
        
        # Report failed points but don't fall back to file downloads
        if failed_points and len(dfs) == 0:
            raise ValueError(
                f"OPeNDAP queries failed for all {len(points)} points. "
                f"No data retrieved. Set allow_fallback=True to enable fallback to file download."
            )
        elif failed_points:
            logger.warning(
                f"OPeNDAP queries failed for {len(failed_points)} out of {len(points)} points. "
                f"Returning data for {len(dfs)} successful points only. "
                f"Set allow_fallback=True to retry failed points via file download."
            )
        
        if not dfs:
            raise ValueError(f"No successful point queries for any of {len(points)} points")
        
        result_df = pd.concat(dfs, ignore_index=False)
        # Rename from OPeNDAP variable name to our standard variable name
        result_df = result_df.rename(columns={variable_opendap: variable_name})
        
        # Apply transformation if defined for this variable
        if variable_name in VARIABLE_TRANSFORMATIONS:
            result_df[variable_name] = result_df[variable_name].apply(VARIABLE_TRANSFORMATIONS[variable_name])
        
        # Apply clipping if specified
        if clip_min is not None or clip_max is not None:
            result_df[variable_name] = result_df[variable_name].clip(lower=clip_min, upper=clip_max)
        
        # Ensure geometry column is at the end if present
        if 'geometry' in result_df.columns:
            cols = [c for c in result_df.columns if c != 'geometry']
            cols.append('geometry')
            result_df = result_df[cols]
        
        return result_df

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
            exclude_values: Any = None,
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
            logger.info(f"reading before granule: {cl.file(before_granule.filename)}")
            t_before = Timer()
            before = before_granule.read(
                variable,
                geometry=geometry,
                resampling=resampling,
                min_value=min_value,
                max_value=max_value,
                exclude_values=exclude_values
            )
            logger.info(f"before granule read complete ({t_before.duration:.2f} seconds)")

            logger.info(f"reading after granule: {cl.file(after_granule.filename)}")
            t_after = Timer()
            after = after_granule.read(
                variable,
                geometry=geometry,
                resampling=resampling,
                min_value=min_value,
                max_value=max_value,
                exclude_values=exclude_values
            )
            logger.info(f"after granule read complete ({t_after.duration:.2f} seconds)")

            logger.info(f"computing temporal interpolation...")
            time_fraction = (time_UTC - before_granule.time_UTC) / (after_granule.time_UTC - before_granule.time_UTC)
            source_diff = after - before
            interpolated_data = before + source_diff * time_fraction
            logger.info(f"GEOS-5 FP interpolation complete ({timer.duration:.2f} seconds total)")


        before_filename = before_granule.filename
        after_filename = after_granule.filename
        filenames = [before_filename, after_filename]
        self.filenames = set(self.filenames) | set(filenames)
        
        if isinstance(interpolated_data, Raster):
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
        return self.query(
            target_variables="SFMC",
            time_UTC=time_UTC,
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
        return self.query(
            target_variables="LAI",
            time_UTC=time_UTC,
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
        return self.query(
            target_variables="LHLAND",
            time_UTC=time_UTC,
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
        return self.query(
            target_variables="EFLUX",
            time_UTC=time_UTC,
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
        return self.query(
            target_variables="PARDR",
            time_UTC=time_UTC,
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
        return self.query(
            target_variables="PARDF",
            time_UTC=time_UTC,
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
        
        return self.query(
            target_variables="AOT",
            time_UTC=time_UTC,
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
        return self.query(
            target_variables="COT",
            time_UTC=time_UTC,
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
        return self.query(
            target_variables="Ts_K",
            time_UTC=time_UTC,
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
        if is_point_geometry(geometry) and ST_K is None:
            return self.query(
                target_variables="Ta_K",
                time_UTC=time_UTC,
                geometry=geometry,
                resampling=resampling
            )
        
        NAME, PRODUCT, VARIABLE = get_variable_info("Ta_K")

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
        return self.query(
            target_variables="Tmin_K",
            time_UTC=time_UTC,
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
        return self.query(
            target_variables="PS",
            time_UTC=time_UTC,
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
        return self.query(
            target_variables="Q",
            time_UTC=time_UTC,
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
                    RH_smooth = self.RH(time_UTC=time_UTC, geometry=geometry, resampling="lanczos")
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
        return self.query(
            target_variables="vapor_kgsqm",
            time_UTC=time_UTC,
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
        return self.query(
            target_variables="ozone_dobson",
            time_UTC=time_UTC,
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
        return self.query(
            target_variables="U2M",
            time_UTC=time_UTC,
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
        return self.query(
            target_variables="V2M",
            time_UTC=time_UTC,
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
        
        return self.query(
            target_variables="CO2SC",
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling,
            expected_hours=EXPECTED_HOURS
        )

    def Ca(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        Atmospheric CO2 concentration in ppm (parts per million by volume)
        Alias for CO2SC.
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of CO2 concentration or DataFrame for point queries
        """
        result = self.CO2SC(time_UTC, geometry=geometry, resampling=resampling)
        
        # Rename CO2SC column to Ca if this is a DataFrame
        if isinstance(result, pd.DataFrame) and 'CO2SC' in result.columns:
            result = result.rename(columns={'CO2SC': 'Ca'})
        # For Raster objects, rename the variable
        elif hasattr(result, 'rename'):
            result = result.rename('Ca')
        
        return result

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
        return self.query(
            target_variables="SWin",
            time_UTC=time_UTC,
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
        return self.query(
            target_variables="SWTDN",
            time_UTC=time_UTC,
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
        return self.query(
            target_variables="ALBVISDR",
            time_UTC=time_UTC,
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
        return self.query(
            target_variables="ALBVISDF",
            time_UTC=time_UTC,
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
        return self.query(
            target_variables="ALBNIRDF",
            time_UTC=time_UTC,
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
        return self.query(
            target_variables="ALBNIRDR",
            time_UTC=time_UTC,
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
        return self.query(
            target_variables="ALBEDO",
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling,
            clip_min=0,
            clip_max=1
        )
    
    def PAR_proportion(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        PAR albedo fraction - the proportion of total albedo from the direct PAR beam component
        Formula: PAR_proportion = ALBVISDR / ALBEDO
        Values range from 0 to 1, indicating what fraction of reflected radiation is in the PAR band
        Use this as a scaling factor to convert total albedo to PAR albedo: albedo_PAR = albedo * PAR_proportion
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of PAR albedo fraction or DataFrame for point queries
        """
        albedo_NWP = self.ALBEDO(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        RVIS_NWP = self.ALBVISDR(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        PAR_proportion = rt.clip(RVIS_NWP / albedo_NWP, 0, 1)
        return PAR_proportion
    
    def NIR_proportion(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, pd.DataFrame]:
        """
        NIR albedo fraction - the proportion of total albedo from the direct NIR beam component
        Formula: NIR_proportion = ALBNIRDR / ALBEDO
        Values range from 0 to 1, indicating what fraction of reflected radiation is in the NIR band
        Use this as a scaling factor to convert total albedo to NIR albedo: albedo_NIR = albedo * NIR_proportion
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of NIR albedo fraction or DataFrame for point queries
        """
        albedo_NWP = self.ALBEDO(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        RNIR_NWP = self.ALBNIRDR(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        NIR_proportion = rt.clip(RNIR_NWP / albedo_NWP, 0, 1)
        return NIR_proportion

    def query(
            self,
            target_variables: Union[str, List[str]] = None,
            targets_df: Union[pd.DataFrame, gpd.GeoDataFrame] = None,
            time_UTC: Union[datetime, str, List[datetime], List[str], pd.Series] = None,
            time_range: Tuple[Union[datetime, str], Union[datetime, str]] = None,
            dataset: str = None,
            geometry: Union[RasterGeometry, Point, MultiPoint, List, gpd.GeoSeries] = None,
            resampling: str = None,
            lat: Union[float, List[float], pd.Series] = None,
            lon: Union[float, List[float], pd.Series] = None,
            dropna: bool = True,
            temporal_interpolation: str = "interpolate",
            variable_name: Union[str, List[str]] = None,
            verbose: bool = False,
            **kwargs) -> Union[Raster, gpd.GeoDataFrame]:
        """
        General-purpose query method that can retrieve any variable from any dataset.
        
        This method provides a flexible interface for retrieving GEOS-5 FP data, supporting both:
        - Raster queries (for spatial data over a region)
        - Point queries (for time-series at specific coordinates using OPeNDAP)
        - Batch spatio-temporal queries (multiple points at different times)
        
        Parameters
        ----------
        target_variables : str or list of str
            Name(s) of the variable(s) to retrieve. Can be either:
            - A single variable name (str): "Ta_K", "SM", "SWin", etc.
            - A list of variable names (List[str]): ["Ta_K", "SM", "SWin"]
            Each can be either:
            - A predefined variable name from GEOS5FP_VARIABLES (e.g., "Ta_K", "SM", "SWin")
            - A raw GEOS-5 FP variable name (e.g., "T2M", "SFMC", "SWGDN")
            When multiple variables are requested with point geometry, each variable becomes a column.
        targets_df : DataFrame or GeoDataFrame, optional
            Input table containing 'time_UTC' and 'geometry' columns. When provided,
            the target variables will be queried for each row and added as new columns
            to this table, which is then returned. This is useful for generating
            validation tables or adding GEOS-5 FP data to existing datasets.
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
        temporal_interpolation : str, default="nearest"
            Method for handling different temporal resolutions when querying multiple variables:
            - "nearest": Use nearest neighbor in time for each variable independently
            - "interpolate": Linear interpolation between observations before and after target time
            This parameter is only used for multi-variable queries at a single time point.
        verbose : bool, optional
            Control logging verbosity. If None (default), uses the connection's verbose setting.
            - True: Detailed console logging with progress information
            - False: Single TQDM progress bar for notebook-friendly display
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
        >>> df = conn.query(
        ...     "T2M",
        ...     time_range=(start_time, end_time),
        ...     dataset="tavg1_2d_slv_Nx",
        ...     lat=34.05,
        ...     lon=-118.25
        ... )
        
        # Example 2: Multiple variables in point query
        >>> df = conn.query(
        ...     ["T2M", "PS", "QV2M"],
        ...     time_range=(start_time, end_time),
        ...     dataset="tavg1_2d_slv_Nx",
        ...     lat=34.05,
        ...     lon=-118.25
        ... )
        
        # Example 3: Raster query at single time
        >>> from rasters import RasterGeometry
        >>> geometry = RasterGeometry.open("target_area.tif")
        >>> raster = conn.query(
        ...     "T2M",
        ...     time_UTC="2024-11-15 12:00:00",
        ...     dataset="tavg1_2d_slv_Nx",
        ...     geometry=geometry
        ... )
        
        # Example 4: Use predefined variable name
        >>> df = conn.query(
        ...     "Ta_K",
        ...     time_range=(start_time, end_time),
        ...     lat=34.05,
        ...     lon=-118.25
        ... )
        
        # Example 5: Query multiple predefined variables
        >>> df = conn.query(
        ...     target_variables=["Ta_K", "SM", "SWin"],
        ...     time_range=(start_time, end_time),
        ...     lat=40.7,
        ...     lon=-74.0
        ... )
        
        # Example 6: Generate table with target variables
        >>> import geopandas as gpd
        >>> targets = gpd.GeoDataFrame({
        ...     'time_UTC': [datetime(2024, 11, 15, 12), datetime(2024, 11, 15, 13)],
        ...     'geometry': [Point(-118.25, 34.05), Point(-74.0, 40.7)]
        ... })
        >>> result = conn.query(
        ...     target_variables=["Ta_C", "RH"],
        ...     targets_df=targets
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
        # Use connection's verbose setting if not explicitly provided
        if verbose is None:
            verbose = self.verbose
        
        # Handle backward compatibility: variable_name is deprecated, use target_variables
        if target_variables is None and variable_name is not None:
            target_variables = variable_name
        elif target_variables is None:
            raise ValueError("target_variables parameter is required")
        
        # Handle targets_df parameter
        if targets_df is not None:
            # Validate targets_df has required columns
            if 'time_UTC' not in targets_df.columns:
                raise ValueError("targets_df must contain 'time_UTC' column")
            if 'geometry' not in targets_df.columns:
                raise ValueError("targets_df must contain 'geometry' column")
            
            # Extract time_UTC and geometry from targets_df
            time_UTC = targets_df['time_UTC']
            geometry = targets_df['geometry']
        
        # Normalize target_variables to list
        if isinstance(target_variables, str):
            variable_names = [target_variables]
            single_variable = True
        else:
            variable_names = target_variables
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
                elif is_point_geometry(geometry):
                    # Single geometry - broadcast to all times
                    geometries = [geometry] * len(times)
                else:
                    raise ValueError("For batch queries with geometry, must provide GeoSeries, list, or single Point/MultiPoint")
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
            
            # Debugging: Log the type and content of geometry
            if verbose:
                logger.info(f"Geometry type: {type(geometry)}")
                logger.info(f"Geometry content: {geometry}")

            # Debugging: Log the parsed times and geometries
            if verbose:
                logger.info(f"Parsed times: {parsed_times}")
                logger.info(f"Parsed geometries: {geometries}")

            # Debugging: Log the coord_to_records mapping
            if verbose:
                logger.info(f"Coordinate to records mapping: {coord_to_records}")

            # Debugging: Log the dataset to variables mapping
            if verbose:
                logger.info(f"Dataset to variables mapping: {dataset_to_variables}")

            # Debugging: Log the results_by_index before returning
            if verbose:
                logger.info(f"Results by index: {results_by_index}")
            
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
            
            # Compute derived/computed variables
            if computed_variables:
                if verbose:
                    logger.info(f"Computing derived variables: {', '.join(computed_variables)}")
                for var_name in computed_variables:
                    try:
                        # Call the appropriate method for each computed variable
                        if var_name == 'RH':
                            # Compute RH from Q, PS, and Ta (for SVP)
                            if 'Q' in result_gdf.columns and 'PS' in result_gdf.columns and 'Ta' in result_gdf.columns:
                                # Import RH calculation utility
                                from GEOS5FP.calculate_RH import calculate_RH
                                
                                # Get the base variables
                                Q = result_gdf['Q'].values
                                PS = result_gdf['PS'].values
                                Ta_K = result_gdf['Ta'].values
                                
                                # Calculate RH
                                RH = calculate_RH(Q, PS, Ta_K)
                                
                                result_gdf['RH'] = RH
                            else:
                                logger.warning("Cannot compute RH: missing required variables (Q, PS, Ta)")
                                result_gdf['RH'] = None
                                
                        elif var_name == 'Ta_C':
                            # Convert Ta from Kelvin to Celsius
                            if 'Ta' in result_gdf.columns:
                                result_gdf['Ta_C'] = result_gdf['Ta'] - 273.15
                            else:
                                logger.warning("Cannot compute Ta_C: missing Ta")
                                result_gdf['Ta_C'] = None
                        
                        elif var_name == 'wind_speed_mps':
                            # Compute wind speed from U2M and V2M components
                            if 'U2M' in result_gdf.columns and 'V2M' in result_gdf.columns:
                                U2M = result_gdf['U2M'].values
                                V2M = result_gdf['V2M'].values
                                result_gdf['wind_speed_mps'] = np.sqrt(U2M**2 + V2M**2)
                            else:
                                logger.warning("Cannot compute wind_speed_mps: missing required variables (U2M, V2M)")
                                result_gdf['wind_speed_mps'] = None
                        
                        elif var_name == 'albedo_visible':
                            # Compute visible albedo scaling factor (ALBEDO / ALBVISDR)
                            if 'ALBEDO' in result_gdf.columns and 'ALBVISDR' in result_gdf.columns:
                                ALBEDO = result_gdf['ALBEDO'].values
                                ALBVISDR = result_gdf['ALBVISDR'].values
                                result_gdf['albedo_visible'] = ALBEDO / ALBVISDR
                            else:
                                logger.warning("Cannot compute albedo_visible: missing required variables (ALBEDO, ALBVISDR)")
                                result_gdf['albedo_visible'] = None
                        
                        elif var_name == 'albedo_NIR':
                            # Compute NIR albedo scaling factor (ALBEDO / ALBNIRDR)
                            if 'ALBEDO' in result_gdf.columns and 'ALBNIRDR' in result_gdf.columns:
                                ALBEDO = result_gdf['ALBEDO'].values
                                ALBNIRDR = result_gdf['ALBNIRDR'].values
                                result_gdf['albedo_NIR'] = ALBEDO / ALBNIRDR
                            else:
                                logger.warning("Cannot compute albedo_NIR: missing required variables (ALBEDO, ALBNIRDR)")
                                result_gdf['albedo_NIR'] = None
                        # Add more computed variables as needed
                    except Exception as e:
                        logger.warning(f"Failed to compute {var_name}: {e}")
                        result_gdf[var_name] = None
                
                # Remove dependency columns that weren't originally requested
                cols_to_keep = ['geometry'] + variable_names
                cols_to_drop = [col for col in result_gdf.columns if col not in cols_to_keep]
                if cols_to_drop:
                    result_gdf = result_gdf.drop(columns=cols_to_drop)
            
            # Ensure geometry column is at the end
            if 'geometry' in result_gdf.columns:
                cols = [c for c in result_gdf.columns if c != 'geometry']
                cols.append('geometry')
                result_gdf = result_gdf[cols]
            
            # Handle targets_df: merge results back into original table
            if targets_df is not None:
                # Drop time_UTC and geometry from result_gdf as they're already in targets_df
                result_cols = [c for c in result_gdf.columns if c not in ['time_UTC', 'geometry']]
                result_data = result_gdf[result_cols].reset_index(drop=True)
                
                # Add variable columns to targets_df
                for col in result_cols:
                    targets_df[col] = result_data[col].values
                
                # Ensure geometry column is at the end
                if 'geometry' in targets_df.columns:
                    cols = [c for c in targets_df.columns if c != 'geometry']
                    cols.append('geometry')
                    targets_df = targets_df[cols]
                
                return targets_df
            
            # Debugging: Log the final result_gdf
            if verbose:
                logger.info(f"Final result_gdf: {result_gdf}")
            
            return result_gdf

        # Handle non-batch queries
        if not is_batch_query:
            if time_UTC is not None:
                # Single time point
                if isinstance(time_UTC, str):
                    time_UTC = parser.parse(time_UTC)
                elif isinstance(time_UTC, list):
                    if len(time_UTC) != 1:
                        raise ValueError("For non-batch queries, time_UTC must be a single value or a single-element list")
                    time_UTC = parser.parse(time_UTC[0]) if isinstance(time_UTC[0], str) else time_UTC[0]

            if geometry is not None:
                if isinstance(geometry, gpd.GeoSeries):
                    if len(geometry) != 1:
                        raise ValueError("For non-batch queries, geometry must be a single value or a single-element GeoSeries")
                    geometry = geometry.iloc[0]
                elif isinstance(geometry, list):
                    if len(geometry) != 1:
                        raise ValueError("For non-batch queries, geometry must be a single value or a single-element list")
                    geometry = geometry[0]

            elif lat is not None and lon is not None:
                # Convert lat/lon to a single Point geometry
                if isinstance(lat, list) or isinstance(lon, list):
                    if len(lat) != 1 or len(lon) != 1:
                        raise ValueError("For non-batch queries, lat and lon must be single values or single-element lists")
                    lat = lat[0] if isinstance(lat, list) else lat
                    lon = lon[0] if isinstance(lon, list) else lon
                geometry = Point(lon, lat)

            # Debugging: Log the single time and geometry
            if verbose:
                logger.info(f"Single time: {time_UTC}")
                logger.info(f"Single geometry: {geometry}")

            # Perform the query for the single time and geometry
            result = self._query_single(
                variable_names=variable_names,
                time_UTC=time_UTC,
                geometry=geometry,
                dataset=dataset,
                resampling=resampling,
                temporal_interpolation=temporal_interpolation,
                verbose=verbose,
                time_range=time_range,
                **kwargs
            )

            # Convert result to appropriate format
            if isinstance(result, pd.DataFrame):
                if 'geometry' in result.columns:
                    result = gpd.GeoDataFrame(result, geometry='geometry', crs='EPSG:4326')
            return result

    def _query_single(self, variable_names, time_UTC, geometry, dataset, resampling, temporal_interpolation, verbose, time_range=None, **kwargs):
        """
        Internal method to handle non-batch queries for a single time and geometry.

        Parameters
        ----------
        variable_names : list of str
            Names of the variables to query.
        time_UTC : datetime
            Single time point for the query.
        geometry : Point or RasterGeometry or MultiPoint
            Single geometry for the query.
        dataset : str
            Dataset name to query from.
        resampling : str
            Resampling method for spatial queries.
        temporal_interpolation : str
            Temporal interpolation method for time queries.
        verbose : bool
            Whether to log detailed information.
        time_range : tuple, optional
            Time range (start_time, end_time) for the query.
        **kwargs : dict
            Additional arguments for the query.

        Returns
        -------
        pd.DataFrame or Raster
            Query result as a DataFrame for point queries or Raster for spatial queries.
        """
        if verbose:
            logger.info(f"Querying dataset: {dataset}")
            logger.info(f"Variables: {variable_names}")
            logger.info(f"Time: {time_UTC}")
            logger.info(f"Geometry: {geometry}")

        # Check if the geometry is a point or raster
        if isinstance(geometry, Point):
            # Handle point query
            if not HAS_OPENDAP_SUPPORT:
                raise ImportError(
                    "Point query support requires xarray and netCDF4. "
                    "Install with: conda install -c conda-forge xarray netcdf4"
                )

            # Perform the point query
            result = self._perform_point_query(
                variable_names=variable_names,
                time_UTC=time_UTC,
                point=geometry,
                dataset=dataset,
                temporal_interpolation=temporal_interpolation,
                time_range=time_range,
                **kwargs
            )

        elif isinstance(geometry, RasterGeometry):
            # Handle raster query
            result = self._perform_raster_query(
                variable_name=variable_names[0],  # Only single variable supported for raster
                time_UTC=time_UTC,
                raster_geometry=geometry,
                dataset=dataset,
                resampling=resampling,
                **kwargs
            )
        elif isinstance(geometry, MultiPoint):
            # Handle MultiPoint geometry by iterating over each point
            if not hasattr(geometry, "geoms"):
                raise ValueError("MultiPoint geometry must have 'geoms' attribute to access individual points")

            results = []
            for point in geometry.geoms:
                if not isinstance(point, Point):
                    raise ValueError("MultiPoint must contain only Point geometries")

                # Perform the point query for each point
                result = self._perform_point_query(
                    variable_names=variable_names,
                    time_UTC=time_UTC,
                    point=point,
                    dataset=dataset,
                    temporal_interpolation=temporal_interpolation,
                    time_range=time_range,
                    **kwargs
                )
                results.append(result)

            # Combine results into a single DataFrame
            result = pd.concat(results, ignore_index=True)
        else:
            raise ValueError("Unsupported geometry type for non-batch query")

        return result

    def _perform_point_query(self, variable_names, time_UTC, point, dataset, temporal_interpolation, time_range=None, **kwargs):
        """
        Internal method to perform a point query for a single point and time.

        Parameters
        ----------
        variable_names : list of str
            Names of the variables to query.
        time_UTC : datetime
            Single time point for the query.
        point : Point
            Shapely Point geometry for the query.
        dataset : str
            Dataset name to query from.
        temporal_interpolation : str
            Temporal interpolation method for time queries.
        time_range : tuple, optional
            Time range (start_time, end_time) for the query.
        **kwargs : dict
            Additional arguments for the query.

        Returns
        -------
        pd.DataFrame
            Query result as a DataFrame.
        """
        if not HAS_OPENDAP_SUPPORT:
            raise ImportError(
                "Point query support requires xarray and netCDF4. "
                "Install with: conda install -c conda-forge xarray netcdf4"
            )

        # Ensure time_UTC or time_range is provided
        if time_UTC is None and time_range is None:
            raise ValueError("Either time_UTC or time_range must be provided for point queries")

        # If time_range is provided, use the start time for the query
        if time_range is not None:
            time_UTC = time_range[0]

        # Parse time_UTC if it's a string
        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)

        logger.info(f"Performing point query at {point} for time {time_UTC}")

        # Implement the actual query logic
        try:
            # Example: Query using xarray and OPeNDAP
            import xarray as xr

            # Open the dataset (replace with actual dataset URL or path)
            dataset_url = self.get_dataset_url(dataset)  # Placeholder for dataset URL retrieval
            ds = xr.open_dataset(dataset_url)

            # Select the variables and time
            ds = ds[variable_names].sel(time=time_UTC, method=temporal_interpolation)

            # Extract the data for the given point
            data = ds.sel(lat=point.y, lon=point.x, method="nearest").to_dataframe()

            # Reset the index and add geometry column
            data = data.reset_index()
            data["geometry"] = point

            return data
        except Exception as e:
            logger.error(f"Failed to query dataset: {e}")
            raise
