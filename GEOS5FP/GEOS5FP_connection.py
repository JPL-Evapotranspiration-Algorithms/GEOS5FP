from contextlib import contextmanager
import errno
import json
import logging
import os
import shutil
import tempfile
from datetime import date, datetime, timedelta, time
from os import makedirs
from os.path import expanduser, exists, join
from pathlib import Path
from time import sleep
from typing import List, Union, Any, Tuple
import posixpath
import colored_logging as cl
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasters as rt
import pooch
import requests
from requests.exceptions import SSLError
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
from .get_variable_info import get_variable_info
from .geometry_utils import is_point_geometry, extract_points
from .query import query as _query_func

# Optional import for point queries via OPeNDAP
try:
    from .GEOS5FP_point import query_geos5fp_point
    HAS_OPENDAP_SUPPORT = True
except ImportError:
    HAS_OPENDAP_SUPPORT = False
    query_geos5fp_point = None

try:
    from filelock import FileLock
except ImportError:
    FileLock = None

logger = logging.getLogger(__name__)


def _safe_remove(path: str) -> bool:
    """Remove a file if it exists, ignoring concurrent deletion races."""
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False
    except OSError as error:
        if error.errno == errno.ENOENT:
            return False
        raise


@contextmanager
def _granule_file_lock(destination_filename: str):
    """Acquire an exclusive inter-process lock for a target granule path."""
    lock_filename = f"{destination_filename}.lock"
    makedirs(os.path.dirname(lock_filename) or ".", exist_ok=True)

    if FileLock is not None:
        with FileLock(lock_filename):
            yield
        return

    with open(lock_filename, "a+") as lock_file:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _publish_staged_file(staged_filename: str, destination_filename: str) -> None:
    try:
        os.replace(staged_filename, destination_filename)
        return
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise

    destination_directory = os.path.dirname(destination_filename) or "."
    descriptor, temporary_filename = tempfile.mkstemp(
        dir=destination_directory,
        prefix=f".{os.path.basename(destination_filename)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as destination, open(staged_filename, "rb") as source:
            shutil.copyfileobj(source, destination)
        shutil.copystat(staged_filename, temporary_filename)
        os.replace(temporary_filename, destination_filename)
    finally:
        _safe_remove(temporary_filename)


class GEOS5FPConnection:
    DEFAULT_URL_BASE = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das"
    DEFAULT_TIMEOUT_SECONDS = 500
    DEFAULT_RETRIES = 3

    query = _query_func

    def __init__(
            self,
            download_directory: str = None,
            remote: str = None,
            save_products: bool = False,
            verbose: bool = True,
            use_opendap: bool = True,
            allow_fallback: bool = False):
        if download_directory is None:
            download_directory = DEFAULT_DOWNLOAD_DIRECTORY

        if remote is None:
            remote = self.DEFAULT_URL_BASE

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
                f"{duration:0.2f}") + f" seconds: {cl.URL(self.remote)}")
        else:
            raise IOError(f"status: {status} URL: {self.remote}")

    def _get_variable_info(self, variable_name: str) -> Tuple[str, str, str]:
        return get_variable_info(variable_name)

    def _is_point_geometry(self, geometry: Any) -> bool:
        return is_point_geometry(geometry)

    def _extract_points(self, geometry: Any) -> List[Tuple[float, float]]:
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
        return posixpath.join(self.remote, f"Y{year:04d}", f"M{month:02d}", f"D{day:02d}") + "/"

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
        return self.dates_available_in_month(available_year, available_month)[-1]

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
        if URL in self._listings:
            return self._listings[URL]
        listing = HTTP_listing(URL, timeout=timeout, retries=retries)
        self._listings[URL] = listing
        return listing

    def list_GEOS5FP_granules(
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
        listing = self.list_remote_directory(day_URL, timeout=timeout, retries=retries)

        if product_name is None:
            URLs = sorted([posixpath.join(day_URL, f) for f in listing if f.endswith(".nc4")])
        else:
            URLs = sorted([posixpath.join(day_URL, f) for f in listing if product_name in f and f.endswith(".nc4")])

        df = pd.DataFrame({"URL": URLs})
        df["time_UTC"] = df["URL"].apply(lambda u: datetime.strptime(posixpath.basename(u).split(".")[4], "%Y%m%d_%H%M"))
        df["product"] = df["URL"].apply(lambda u: posixpath.basename(u).split(".")[3])
        return df[["time_UTC", "product", "URL"]]

    def generate_filenames(
            self,
            date_UTC: Union[datetime, str],
            product_name: str,
            interval: int,
            expected_hours: List[float] = None) -> pd.DataFrame:
        if isinstance(date_UTC, str):
            date_UTC = parser.parse(date_UTC).date()

        if expected_hours is None:
            if product_name.startswith("tavg3"):
                expected_hours = [1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5]
            elif interval == 1:
                expected_hours = np.arange(0.5, 24.5, 1)
            elif interval == 3:
                expected_hours = np.arange(0.0, 24.0, 3)
            else:
                raise ValueError(f"unrecognized GEOS-5 FP interval: {interval}")

        rows = []
        expected_times = (
            [datetime.combine(date_UTC - timedelta(days=1), time(0)) + timedelta(hours=float(expected_hours[-1]))]
            + [datetime.combine(date_UTC, time(0)) + timedelta(hours=float(hour)) for hour in expected_hours]
            + [datetime.combine(date_UTC + timedelta(days=1), time(0)) + timedelta(hours=float(expected_hours[0]))]
        )

        for time_UTC in expected_times:
            filename = f"GEOS.fp.asm.{product_name}.{time_UTC:%Y%m%d_%H%M}.V01.nc4"
            day_URL = self.day_URL(time_UTC.date())
            URL = posixpath.join(day_URL, filename)
            rows.append([time_UTC, URL])

        return pd.DataFrame(rows, columns=["time_UTC", "URL"])

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
        return join(download_directory, posixpath.basename(URL))

    def download_file(self, URL: str, filename: str = None, retries: int = 3, wait_seconds: int = 30) -> 'GEOS5FPGranule':
        from .exceptions import GEOS5FPDayNotAvailable, GEOS5FPGranuleNotAvailable, FailedGEOS5FPDownload, GEOS5FPSSLError
        from .validate_GEOS5FP_NetCDF_file import validate_GEOS5FP_NetCDF_file

        if filename is None:
            filename = self.download_filename(URL)

        expanded_filename = os.path.abspath(os.path.expanduser(filename))

        with _granule_file_lock(expanded_filename):
            if exists(expanded_filename):
                logger.info(f"checking existing file: {filename}")
                validation_result = validate_GEOS5FP_NetCDF_file(expanded_filename, verbose=False)

                if validation_result.is_valid:
                    logger.info(f"existing file is valid: {filename} ({validation_result.metadata.get('file_size_mb', 'unknown')} MB)")
                    return GEOS5FPGranule(filename)
                else:
                    logger.warning(f"existing file is invalid, removing: {filename}")
                    for error in validation_result.errors[:3]:
                        logger.warning(f"  validation error: {error}")
                    try:
                        if _safe_remove(expanded_filename):
                            logger.info(f"removed invalid file: {filename}")
                    except OSError as e:
                        logger.warning(f"failed to remove invalid file {filename}: {e}")

            try:
                with create_robust_session() as session:
                    head_resp = session.head(URL, timeout=30, verify=True)
                if head_resp.status_code != 404:
                    head_resp.raise_for_status()
            except SSLError as e:
                logger.error(f"SSL connection failed for URL: {URL}")
                raise GEOS5FPSSLError("Failed to establish SSL connection", original_error=e, url=URL)
            except Exception as e:
                logger.error(f"Failed to check remote file existence: {e}")
                raise FailedGEOS5FPDownload(f"Connection error: {e}")

            if head_resp.status_code == 404:
                directory_URL = posixpath.dirname(URL)
                try:
                    with create_robust_session() as session:
                        dir_resp = session.head(directory_URL, timeout=30, verify=True)
                    if dir_resp.status_code != 404:
                        dir_resp.raise_for_status()
                except SSLError as e:
                    raise GEOS5FPSSLError("Failed to establish SSL connection to directory", original_error=e, url=directory_URL)
                except Exception as e:
                    raise FailedGEOS5FPDownload(f"Connection error checking directory: {e}")
                if dir_resp.status_code == 404:
                    raise GEOS5FPDayNotAvailable(directory_URL)
                raise GEOS5FPGranuleNotAvailable(URL)

            download_root = Path(os.path.expanduser(self.download_directory)).resolve()
            staging_root = Path(pooch.os_cache("GEOS5FP")).expanduser().resolve()
            if staging_root.is_relative_to(download_root):
                raise FailedGEOS5FPDownload(
                    f"Pooch staging directory must be outside the download directory: {staging_root}"
                )
            makedirs(staging_root, exist_ok=True)
            makedirs(os.path.dirname(expanded_filename) or ".", exist_ok=True)

            download_attempts = 0
            max_download_attempts = retries
            downloader = pooch.HTTPDownloader(
                progressbar=True,
                chunk_size=1024 * 1024,
                timeout=120,
            )

            while download_attempts < max_download_attempts:
                download_attempts += 1
                logger.info(f"download attempt {download_attempts}/{max_download_attempts}: {URL}")

                try:
                    with tempfile.TemporaryDirectory(dir=staging_root) as staging_directory:
                        staged_filename = os.path.join(
                            staging_directory,
                            os.path.basename(expanded_filename),
                        )
                        if Path(staging_directory).resolve().is_relative_to(download_root):
                            raise FailedGEOS5FPDownload(
                                "Pooch staging directory must be outside the download directory"
                            )
                        downloader(
                            url=URL,
                            output_file=staged_filename,
                            pooch=None,
                        )
                        if not exists(staged_filename) or os.path.getsize(staged_filename) == 0:
                            raise IOError(f"zero-size or missing file from download: {URL}")
                        _publish_staged_file(staged_filename, expanded_filename)
                    result_filename = filename
                except SSLError as e:
                    if download_attempts >= max_download_attempts:
                        raise GEOS5FPSSLError("Failed to establish SSL connection", original_error=e, url=URL)
                    logger.warning(f"download attempt {download_attempts} failed: {e}")
                    sleep(wait_seconds)
                    continue
                except FailedGEOS5FPDownload:
                    if download_attempts >= max_download_attempts:
                        raise
                    sleep(wait_seconds)
                    continue
                except Exception as e:
                    if download_attempts >= max_download_attempts:
                        raise FailedGEOS5FPDownload(str(e))
                    logger.warning(f"download attempt {download_attempts} failed: {e}")
                    sleep(wait_seconds)
                    continue

                logger.info(f"validating downloaded file: {result_filename}")
                validation_result = validate_GEOS5FP_NetCDF_file(expanded_filename, verbose=False)

                if validation_result.is_valid:
                    logger.info(f"download and validation successful: {result_filename} ({validation_result.metadata.get('file_size_mb', 'unknown')} MB)")
                    return GEOS5FPGranule(result_filename)
                else:
                    logger.warning(f"downloaded file failed validation: {result_filename}")
                    for error in validation_result.errors[:3]:
                        logger.warning(f"  validation error: {error}")

                    try:
                        if _safe_remove(expanded_filename):
                            logger.info(f"removed invalid downloaded file: {result_filename}")
                    except OSError as e:
                        logger.warning(f"failed to remove invalid file {result_filename}: {e}")

                    if download_attempts >= max_download_attempts:
                        error_summary = '; '.join(validation_result.errors[:2])
                        raise FailedGEOS5FPDownload(f"downloaded file validation failed after {max_download_attempts} attempts: {error_summary}")

                    sleep(wait_seconds)

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

        search_date = time_UTC.date()
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

        after_time_UTC, after_URL = after_listing.iloc[0][["time_UTC", "URL"]]
        before_granule = self.download_file(before_URL)
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
            clip_max: Any = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        if isinstance(time_UTC, (pd.Series, pd.DatetimeIndex, list, tuple, np.ndarray)):
            time_series = time_UTC if isinstance(time_UTC, pd.Series) else pd.Series(time_UTC)

            if hasattr(geometry, '__len__') and not isinstance(geometry, (str, RasterGeometry)):
                if len(time_series) != len(geometry):
                    raise ValueError(
                        f"Length mismatch: time_UTC has {len(time_series)} values "
                        f"but geometry has {len(geometry)} elements"
                    )

                results = [
                    self.variable(
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
                    for time_val, geom_val in zip(time_series, geometry)
                ]

                if all(isinstance(r, pd.DataFrame) for r in results):
                    return pd.concat(results, ignore_index=True)
                elif all(isinstance(r, (Raster, np.ndarray)) for r in results):
                    return np.array(results)
                return results
            else:
                unique_times = time_series.unique()
                time_to_result = {
                    unique_time: self.variable(
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
                    for unique_time in unique_times
                }
                results = [time_to_result[t] for t in time_series]
                if all(isinstance(r, pd.DataFrame) for r in results):
                    return pd.concat(results, ignore_index=True)
                elif all(isinstance(r, (Raster, np.ndarray)) for r in results):
                    return np.array(results)
                return results

        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)

        NAME, PRODUCT, VARIABLE = get_variable_info(variable_name)

        if is_point_geometry(geometry):
            if self.use_opendap and HAS_OPENDAP_SUPPORT:
                try:
                    return self._query_point_via_opendap(
                        NAME, PRODUCT, VARIABLE, variable_name,
                        time_UTC, geometry, resampling, interval,
                        expected_hours, min_value, max_value,
                        exclude_values, cmap, clip_min, clip_max
                    )
                except Exception as e:
                    if not self.allow_fallback:
                        raise

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
        if not HAS_OPENDAP_SUPPORT:
            raise ImportError(
                "Point query support requires xarray and netCDF4. "
                "Install with: conda install -c conda-forge xarray netcdf4"
            )

        points = extract_points(geometry)
        dfs = []
        variable_opendap = VARIABLE.lower()
        failed_points = []

        for lat, lon in points:
            try:
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
                    failed_points.append((lat, lon))
                    continue

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

        if not dfs:
            raise ValueError(f"No successful point queries for any of {len(points)} points")

        result_df = pd.concat(dfs, ignore_index=False)
        result_df = result_df.rename(columns={variable_opendap: variable_name})

        if variable_name in VARIABLE_TRANSFORMATIONS:
            result_df[variable_name] = result_df[variable_name].apply(VARIABLE_TRANSFORMATIONS[variable_name])

        if clip_min is not None or clip_max is not None:
            result_df[variable_name] = result_df[variable_name].clip(lower=clip_min, upper=clip_max)

        if 'geometry' in result_df.columns:
            cols = [c for c in result_df.columns if c != 'geometry'] + ['geometry']
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
            exclude_values=None,
            interval: int = None,
            expected_hours: List[float] = None,
            timeout: float = None,
            retries: int = None,
            use_http_listing: bool = DEFAULT_USE_HTTP_LISTING) -> Raster:
        if interval is None or (expected_hours is None and product.startswith("tavg3")):
            if product.startswith("tavg3"):
                interval = 3
                if expected_hours is None:
                    expected_hours = [1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5]
            elif product.startswith("inst3"):
                interval = 3
            elif product.startswith(("tavg1", "inst1")):
                interval = 1

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
        interpolated_data = before + (after - before) * time_fraction

        filenames = [before_granule.filename, after_granule.filename]
        self.filenames = set(self.filenames) | set(filenames)

        if isinstance(interpolated_data, Raster):
            interpolated_data["filenames"] = filenames

        if cmap is not None:
            interpolated_data.cmap = cmap

        return interpolated_data

    def SFMC(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
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

    def LAI(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
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
        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)
        LAI = self.LAI(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        return rt.clip(1.05 - np.exp(-0.5 * LAI), 0, 1)

    def LHLAND(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(
            target_variables="LHLAND",
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling,
            interval=1,
            exclude_values=[1.e+15]
        )

    def EFLUX(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(
            target_variables="EFLUX",
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling,
            interval=1
        )

    def PARDR(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(
            target_variables="PARDR",
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling,
            clip_min=0
        )

    def PARDF(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(
            target_variables="PARDF",
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling,
            clip_min=0
        )

    def AOT(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        EXPECTED_HOURS = [1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5]
        return self.query(
            target_variables="AOT",
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling,
            expected_hours=EXPECTED_HOURS
        )

    def COT(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(
            target_variables="COT",
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling
        )

    def Ts_K(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
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
            return_scale_and_bias: bool = False) -> Union[Raster, np.ndarray, pd.DataFrame]:
        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)

        if is_point_geometry(geometry) and ST_K is None:
            return self.query(
                target_variables="Ta_K",
                time_UTC=time_UTC,
                geometry=geometry,
                resampling=resampling
            )

        NAME, PRODUCT, VARIABLE = get_variable_info("Ta_K")

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

    def Tmin_K(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(
            target_variables="Tmin_K",
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling
        )

    def SVP_Pa(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        Ta_C = self.Ta_C(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        return 0.6108 * np.exp((17.27 * Ta_C) / (Ta_C + 237.3)) * 1000

    def SVP_kPa(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        return self.SVP_Pa(time_UTC=time_UTC, geometry=geometry, resampling=resampling) / 1000

    def Ta_C(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        return self.Ta_K(time_UTC=time_UTC, geometry=geometry, resampling=resampling) - 273.15

    def Tmin_C(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        return self.Tmin_K(time_UTC=time_UTC, geometry=geometry, resampling=resampling) - 273.15

    def PS(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(target_variables="PS", time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    def Q(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(target_variables="Q", time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    def Ea_Pa(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        RH = self.RH(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        SVP_Pa = self.SVP_Pa(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        return RH * SVP_Pa

    def Td_K(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        Ta_K = self.Ta_K(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        RH = self.RH(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        return Ta_K - (100.0 - RH * 100.0) / 5.0

    def RH(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        Q = self.Q(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        PS = self.PS(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        Ta_K = self.Ta_K(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        Ta_C = Ta_K - 273.15
        SVP_Pa = 611.2 * np.exp((17.67 * Ta_C) / (Ta_C + 243.5))
        Mw = 18.015268
        Md = 28.96546e-3
        epsilon = Mw / (Md * 1000)
        w = Q / (1 - Q)
        ws = epsilon * SVP_Pa / (PS - SVP_Pa)
        return rt.clip(w / ws, 0, 1)

    def vapor_kgsqm(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(target_variables="vapor_kgsqm", time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    def vapor_gccm(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(target_variables="vapor_gccm", time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    def ozone_dobson(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(target_variables="ozone_dobson", time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    def ozone_cm(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(target_variables="ozone_cm", time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    def U2M(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(target_variables="U2M", time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    def V2M(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(target_variables="V2M", time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    def wind_speed_mps(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        U2M = self.U2M(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        V2M = self.V2M(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        return np.sqrt(U2M ** 2 + V2M ** 2)

    def CO2SC(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(target_variables="CO2SC", time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    Ca = CO2SC

    def SWin(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(target_variables="SWin", time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    def SWTDN(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(target_variables="SWTDN", time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    def ALBVISDR(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(target_variables="ALBVISDR", time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    def ALBVISDF(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(target_variables="ALBVISDF", time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    def ALBNIRDF(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(target_variables="ALBNIRDF", time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    def ALBNIRDR(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(target_variables="ALBNIRDR", time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    def ALBEDO(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        return self.query(target_variables="ALBEDO", time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    def albedo_visible(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        """
        visible surface albedo (ALBEDO * (ALBVISDR / ALBEDO) = ALBVISDR)
        """
        return self.ALBVISDR(time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    def albedo_NIR(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Union[Raster, np.ndarray, pd.DataFrame]:
        """
        near-infrared surface albedo (ALBEDO * (ALBNIRDR / ALBEDO) = ALBNIRDR)
        """
        return self.ALBNIRDR(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
