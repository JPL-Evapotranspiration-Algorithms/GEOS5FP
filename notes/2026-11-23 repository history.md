# History and Evolution of the `GEOS5FP` Python Package

**Author:** Gregory H. Halverson (NASA Jet Propulsion Laboratory)  
**Repository:** [JPL-Evapotranspiration-Algorithms/GEOS5FP](https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP)  
**Timeline Covered:** June 25, 2024 (`v1.0.1`) – November 17, 2026 (`v2.24.0`)  

---

## Overview

The `GEOS5FP` Python package provides programmatic access, spatio-temporal interpolation, downscaling, and tabular extraction for NASA's **GEOS-5 Forward Processing (GEOS-5 FP)** near-real-time meteorological assimilation products.

Across 42 releases spanning over two years, the codebase evolved from a single-file raster download-and-interpolation script into a modular, multi-modal scientific data library supporting:
- Gridded raster retrieval and thermal/moisture downscaling
- High-throughput OPeNDAP point and spatio-temporal table queries
- Derived meteorological variable computation
- Subprocess-isolated NetCDF4/HDF5 validation and staged HTTP downloads

---

## Major Eras of Development

### Era 1: Monolithic Prototype & Initial PyPI Release (`v1.0.1` – `v1.0.7`, June 2024 – March 2025)
- **Initial Commit (`June 25, 2024`):** Started as a lowercase package (`geos5fp`) containing a 1,774-line `geos5fp/geos5fp.py` file and `geos5fp/downscaling/downscaling.py`.
- **Core Capabilities at Launch:**
  - HTTP directory scraping (`BeautifulSoup`) against NASA NCCS (`portal.nccs.nasa.gov/datashare/gmao/geos-fp/das`).
  - Granule retrieval via external `wget` system calls.
  - Linear temporal interpolation between bracketing forecast/assimilation granules (`before_and_after` and `interpolate`).
  - Spatial downscaling (`linear_downscale`, `bias_correct`, `downscale_air_temperature`, `downscale_soil_moisture`, `downscale_vapor_pressure_deficit`, and `downscale_relative_humidity`).
- **Modularization & CI (`v1.0.2` – `v1.0.7`):**
  - Split `geos5fp.py` into `GEOS5FPConnection`, `GEOS5FPGranule`, `HTTP_listing`, `constants`, `exceptions`, and an internal `Timer` utility.
  - Replaced external `cl`/`timer` dependencies with `colored-logging` and internal timing.
  - Added GitHub Actions CI workflows and dependency import tests (`test_import_dependencies.py`).

### Era 2: Local Caching, Uppercase Rebranding & Pure-Python Downloads (`v1.1.1` – `v1.2.4`, March 2025 – October 2025)
- **Local Cache Checks & Package Rename (`v1.1.1` – `v1.1.2`):**
  - Short-circuited remote HTTP checks when valid granules already existed on disk.
  - Eliminated intermediate GeoTIFF product directories (`GEOS5FP_products`) and working directories in favor of a single `download_directory` (`~/data/GEOS5FP`).
  - Renamed the package and module from `geos5fp` to uppercase `GEOS5FP` to align with NASA/JPL algorithm naming conventions.
- **Pure-Python Streaming Downloads (`v1.2.0` – `v1.2.4`):**
  - Removed the system dependency on `wget` (after a brief experiment with `earthaccess`), replacing it with chunked `requests` streaming and `tqdm` progress bars.
  - Switched package versioning to dynamic `importlib.metadata.version("GEOS5FP")`.
  - Added fast nearest-neighbor point extraction when reading granules with `Point` geometries (`v1.2.4`).

### Era 3: Crash-Resilient NetCDF Validation & SSL Hardening (`v1.3.0` – `v1.3.2`, October 2025 – November 2025)
- **Comprehensive File Validation (`v1.3.0`):**
  - Introduced `validate_GEOS5FP_NetCDF_file.py` to check file existence, byte size thresholds, filename regex structure (`GEOS.fp.asm.<product>.<YYYYMMDD_HHMM>.V01.nc4`), CRS/bounds, variable presence, and NaN ratios.
  - Added **subprocess isolation** (`_validate_in_subprocess`) with JSON IPC and configurable timeouts so low-level C++ `SIGABRT` or segmentation faults in GDAL/HDF5 when opening truncated downloads would not terminate the parent Python pipeline.
- **SSL/TLS Fallback Strategies (`v1.3.1` – `v1.3.2`):**
  - Added `GEOS5FPSSLError`, `create_robust_session`, `create_legacy_ssl_context`, and `make_head_request_with_ssl_fallback` to handle `UNEXPECTED_EOF_WHILE_READING` and legacy cipher negotiation issues with NASA NCCS servers.

### Era 4: OPeNDAP Integration, Vectorized Tables & Computed Variables (`v2.0.0` – `v2.11.0`, November 2025 – December 2025)
- **Major Architecture Shift (`v2.0.0` – `v2.1.0`):**
  - Added `GEOS5FP_point.py` using `xarray` and `netCDF4` to query NASA's OPeNDAP server (`opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim`) directly for point and time-series data without downloading full global `.nc4` granules.
  - Centralized variable definitions into an embedded `GEOS5FP/variables.csv` lookup table (`_load_variables()`).
  - Implemented **vectorized spatio-temporal table queries** (`query()` with `targets_df`): grouping target records by unique `(lat, lon)` coordinates, clustering timestamps (`cluster_times.py`) to keep OPeNDAP slices small, and batching multi-variable requests per dataset (`query_geos5fp_point_multi`).
  - Added automatic derivation of **computed meteorological variables** (`RH`, `Ta_C`, `Ea_Pa`, `SVP_Pa`, `VPD_kPa`, `Td_K`, `wind_speed_mps`) and built a comprehensive unit test suite (`tests/`).
- **Performance & Reliability Tuning (`v2.3.0` – `v2.11.0`):**
  - Added elapsed time and ETA progress tracking for multi-batch queries (`v2.3.0`).
  - Disabled xarray CF decoding/locking on remote OPeNDAP datasets and added socket timeouts + retry backoff to resolve remote OPeNDAP hangs (`v2.4.0`, `v2.9.1`).
  - Added surface albedo proportion and broadband albedo variables (`PAR_proportion`, `NIR_proportion`, `albedo_visible`, `albedo_NIR`) (`v2.5.0` – `v2.6.0`).
  - Added support for `pandas.Series`, `DatetimeIndex`, lists, tuples, and NumPy arrays passed as `time_UTC` (`v2.7.0` – `v2.8.0`).
  - Implemented 7-day temporal chunking for long single-point time-series queries (`v2.10.0` – `v2.11.0`).

### Era 5: Query Modularization, Pooch Downloads & GDAL Independence (`v2.12.0` – `v2.24.0`, December 2025 – November 2026)
- **Standalone Query Engine & Return-Type Standardization (`v2.12.0` – `v2.21.0`):**
  - Extracted the 1,000+ line `query()` method out of `GEOS5FP_connection.py` into its own `GEOS5FP/query.py` module (`v2.12.0`).
  - Standardized single-variable non-raster queries to return 1D `numpy.ndarray` values (`v2.13.0`), added unwrapping for `rasters` point geometry wrappers (`v2.14.0` – `v2.15.0`), and restored raster snapshot routing (`v2.20.0`).
  - Added `Tmin_C`, `PAR_albedo`, and `NIR_albedo`, and fixed 3-hourly `tavg3` expected hour intervals (`v2.18.0` – `v2.21.0`).
- **Pooch Staged Downloads & Cloud-Sync Safety (`v2.22.0` – `v2.23.0`):**
  - Replaced custom `requests` chunk loops with `pooch.HTTPDownloader` (`v2.22.0`), staging network transfers in `pooch.os_cache("GEOS5FP")` outside the target directory before atomically publishing via `os.replace` (with cross-device fallback `_publish_staged_file`). This resolved file-write timeouts when `~/data` resided on cloud-synced storage such as OneDrive.
  - Enabled Pooch download progress bars (`v2.23.0`).
- **GDAL HDF5 Fallback & Concurrency Locking (`v2.24.0`):**
  - Added direct `netCDF4` fallback reading (`GEOS5FPGranule._read_via_netcdf4`) and validation (`_validate_with_netcdf4`) so granules load reliably even in environments where GDAL lacks `gdal_HDF5.so`.
  - Added inter-process file locking (`_granule_file_lock`) around granule validation and download to prevent concurrent worker races, and updated minimum Python support to `>=3.11`.

---

## Complete Release Timeline

| Version | Release Date (UTC) | Summary of Changes |
| :--- | :--- | :--- |
| **`v1.0.1`** | 2024-07-20 | Initial PyPI release (`geos5fp`); HTTP directory listing, `wget` download, temporal interpolation, and spatial downscaling. |
| **`v1.0.2`** | 2025-02-26 | Modularized monolithic script into `GEOS5FPConnection`, `GEOS5FPGranule`, `HTTP_listing`, `constants`, and `exceptions`. |
| **`v1.0.3`** | 2025-02-26 | Fixed relative imports, added dependency import tests, and configured GitHub Actions CI. |
| **`v1.0.4`** | 2025-02-26 | Exported `constants` and `exceptions` at package top level. |
| **`v1.0.5`** | 2025-02-28 | Standardized module-level loggers across connection and granule classes. |
| **`v1.0.6`** | 2025-03-10 | Updated README installation instructions for `wget`. |
| **`v1.0.7`** | 2025-03-10 | Added `brew` and `mamba` installation instructions for `wget`. |
| **`v1.1.1`** | 2025-03-28 | Renamed package to uppercase `GEOS5FP`; skipped remote checks when local files exist; removed intermediate product files. |
| **`v1.1.2`** | 2025-07-15 | Reduced redundant remote HTTP `HEAD` requests when granules are already cached locally. |
| **`v1.2.0`** | 2025-07-22 | Removed `working_directory`; defaulted `download_directory` to `~/data/GEOS5FP`; replaced `wget` with `requests` + `tqdm`; improved date/time string parsing. |
| **`v1.2.1`** | 2025-07-22 | Cleaned up unused constant imports in `GEOS5FP_granule.py`. |
| **`v1.2.2`** | 2025-07-22 | Reduced verbose directory initialization logging. |
| **`v1.2.3`** | 2025-08-04 | Added `ChunkedEncodingError` and `ConnectionError` retry handling; switched to `importlib.metadata` versioning. |
| **`v1.2.4`** | 2025-10-24 | Added nearest-neighbor resampling when reading granules for `Point` geometries. |
| **`v1.3.0`** | 2025-10-27 | Added comprehensive NetCDF validation (`validate_GEOS5FP_NetCDF_file`) with subprocess isolation against C++/GDAL crashes. |
| **`v1.3.1`** | 2025-10-28 | Added `GEOS5FPSSLError` and automatic SSL verification fallback for NCCS connections. |
| **`v1.3.2`** | 2025-11-05 | Added multi-stage SSL fallback including legacy `SECLEVEL=1` cipher context. |
| **`v2.0.0`** | 2025-11-24 | Major release: OPeNDAP point/time-series queries (`GEOS5FP_point.py`), CSV-driven variable registry (`variables.csv`), vectorized table queries, and unit test suite. |
| **`v2.0.1`** | 2025-11-25 | Suppressed `NumExpr` thread-count log noise on package import. |
| **`v2.1.0`** | 2025-11-25 | Added `targets_df` table enrichment, `cluster_times`, `calculate_RH`, computed variables (`RH`, `Ta_C`, etc.), SSL helper modules, and `lanczos` default resampling. |
| **`v2.3.0`** | 2025-11-25 | Added elapsed time and ETA progress logging/bars for batch and point queries (includes `v2.2.0` work). |
| **`v2.4.0`** | 2025-12-10 | Added `Ca` alias, `wind_speed_mps`, and `decode_cf=False` OPeNDAP optimization to eliminate remote hangs. |
| **`v2.5.0`** | 2025-12-16 | Added visible/NIR albedo scale factors, examples, and expanded `README.md` variable documentation. |
| **`v2.6.0`** | 2025-12-16 | Renamed visible albedo fraction to `PAR_proportion` alongside `NIR_proportion`. |
| **`v2.7.0`** | 2025-12-18 | Added support for passing `pandas.Series`, lists, tuples, and NumPy arrays as `time_UTC`. |
| **`v2.8.0`** | 2025-12-18 | Added `pd.DatetimeIndex` support and suppressed timezone/NumPy warnings in tests. |
| **`v2.9.0`** | 2025-12-19 | Added detailed per-step progress and ETA logging for granule searches and series queries. |
| **`v2.9.1`** | 2025-12-22 | Added socket timeouts and retry backoff to OPeNDAP point queries; refined progress bar descriptions. |
| **`v2.10.0`** | 2025-12-23 | Unified variable retrieval methods under `.query()` and added single-point geometry broadcasting. |
| **`v2.11.0`** | 2025-12-30 | Implemented 7-day temporal chunking for long single-point OPeNDAP time-series queries. |
| **`v2.12.0`** | 2026-01-02 | Extracted `query()` into standalone `GEOS5FP/query.py` module. |
| **`v2.13.0`** | 2026-01-02 | Standardized single-variable non-raster queries to return `numpy.ndarray`. |
| **`v2.14.0`** | 2026-01-02 | Added unwrapping for `rasters` point wrapper objects in batch queries. |
| **`v2.15.0`** | 2026-01-14 | Extended `rasters` point unwrapping to single-time point queries. |
| **`v2.16.0`** | 2026-01-15 | Temporarily disabled `tqdm` notebook progress bars to prevent UI freezing. |
| **`v2.17.0`** | 2026-01-16 | Reduced batch time-clustering window to 1 day and fixed batch count estimation for multi-dataset queries. |
| **`v2.18.0`** | 2026-01-16 | Added `Tmin_C` computed variable. |
| **`v2.19.0`** | 2026-01-20 | Added `PAR_albedo` and `NIR_albedo` methods and constants. |
| **`v2.20.0`** | 2026-07-23 | Restored single-time raster snapshot routing in `GEOS5FPConnection.query()`. |
| **`v2.21.0`** | 2026-10-12 | Fixed default `expected_hours` (`[1.5, 4.5, ..., 22.5]`) for 3-hourly `tavg3` products. |
| **`v2.22.0`** | 2026-11-10 | Migrated file downloads to `pooch` with out-of-tree temporary staging and atomic publication. |
| **`v2.23.0`** | 2026-11-10 | Enabled `pooch` download progress bar and added `examples/download_progress.py`. |
| **`v2.24.0`** | 2026-11-17 | Added `netCDF4` fallback when GDAL HDF5 plugin is unavailable, inter-process file locking (`_granule_file_lock`), and `albedo_visible`/`albedo_NIR` computed variables; set minimum Python to `>=3.11`. |