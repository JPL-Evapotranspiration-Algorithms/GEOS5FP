# Comprehensive Development Plan: `GEOS5FP` (`v1.0.1` – `v2.25.0`)

**Author:** Gregory H. Halverson (NASA Jet Propulsion Laboratory)  
**Repository:** [JPL-Evapotranspiration-Algorithms/GEOS5FP](https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP)  
**Date:** November 23, 2026  
**Current Release:** `v2.24.0`  
**Target Release:** `v2.25.0`  

---

## Executive Summary

The `GEOS5FP` Python package provides programmatic retrieval, spatio-temporal interpolation, thermal/moisture downscaling, and tabular extraction for NASA's **GEOS-5 Forward Processing (GEOS-5 FP)** near-real-time meteorological assimilation products.

Over 42 releases from `v1.0.1` (July 2024) through `v2.24.0` (November 2026), `GEOS5FP` evolved from a single-file raster download-and-interpolation script into a modular, multi-modal scientific data access engine supporting:

- **Gridded raster retrieval and downscaling** via `rasters`, `netCDF4`, and GDAL
- **High-throughput OPeNDAP point, multi-point, time-series, and batch table queries** via `xarray`, `netCDF4`, `pandas`, and `geopandas`
- **Automatic evaluation of derived meteorological variables** (`RH`, `Ta_C`, `wind_speed_mps`, `VPD_kPa`, etc.)
- **Subprocess-isolated NetCDF4 validation and staged Pooch downloads** resilient to C++ library crashes and cloud-synced filesystems (e.g., OneDrive)

**Version `2.25.0`** will be a comprehensive stabilization, feature-parity, and repository-hygiene release that resolves all **12 open GitHub issues (`#60`–`#71`)** in a single coordinated pass—unblocking downstream pipelines (`BESS-JPL`, `JET3`, `ECOv003-L3T-L4T-JET`) and establishing uniform behavior across gridded, point, time-series, and batch query modes.

---

## Part I: Retrospective — Past Development (`v1.0.1` – `v2.24.0`)

### Era 1: Monolithic Prototype & Initial PyPI Release (`v1.0.1` – `v1.0.7`, June 2024 – March 2025)

The initial architecture focused on spatial and temporal interpolation of GEOS-5 FP granules onto target satellite grids (such as Landsat and ECOSTRESS tiles).

- **`v1.0.1` (2024-07-20):** Initial PyPI release (`geos5fp`), containing a single 1,774-line `geos5fp/geos5fp.py` module and `geos5fp/downscaling/downscaling.py`. Implemented HTTP directory scraping (`BeautifulSoup`) against NASA NCCS (`portal.nccs.nasa.gov/datashare/gmao/geos-fp/das`), `wget` downloads, linear temporal interpolation (`before_and_after` / `interpolate`), and empirical spatial downscaling (`linear_downscale`, `bias_correct`, `downscale_air_temperature`, `downscale_soil_moisture`, `downscale_vapor_pressure_deficit`, and `downscale_relative_humidity`).
- **`v1.0.2` (2025-02-26):** Decomposed the monolithic script into distinct modules: `GEOS5FPConnection`, `GEOS5FPGranule`, `HTTP_listing`, `constants`, `exceptions`, and an internal performance `Timer`. Set minimum Python version.
- **`v1.0.3` (2025-02-26):** Fixed relative imports, added `tests/test_import_dependencies.py` and `tests/test_import_geos5fp.py`, and configured GitHub Actions CI.
- **`v1.0.4` (2025-02-26):** Exported `constants` and `exceptions` at the package top level.
- **`v1.0.5` (2025-02-28):** Standardized module-level loggers across connection and granule classes.
- **`v1.0.6` – `v1.0.7` (2025-03-10):** Updated `README.md` with `brew` and `mamba` installation instructions for `wget`.

---

### Era 2: Local Cache Optimization, Uppercase Rebranding & Pure-Python Downloads (`v1.1.1` – `v1.2.4`, March 2025 – October 2025)

Focus shifted toward eliminating external CLI tool dependencies (`wget`), simplifying disk storage, and reducing unnecessary network round-trips.

- **`v1.1.1` (2025-03-28):** Renamed package and module from `geos5fp` to uppercase `GEOS5FP` to align with NASA/JPL algorithm naming conventions. Added local file existence checks before hitting remote servers and eliminated intermediate GeoTIFF product caches (`GEOS5FP_products`) and `working_directory`.
- **`v1.1.2` (2025-07-15):** Reduced redundant HTTP `HEAD` requests when granules were already present on disk.
- **`v1.2.0` (2025-07-22):** Defaulted `download_directory` to `~/data/GEOS5FP`, replaced `wget` with pure-Python chunked `requests` streaming and `tqdm` progress bars, and added flexible string date/time parsing (`dateutil.parser`) across all variable methods.
- **`v1.2.1` – `v1.2.2` (2025-07-22):** Cleaned up unused imports in `GEOS5FP_granule.py` and reduced verbose initialization logging.
- **`v1.2.3` (2025-08-04):** Added `ChunkedEncodingError` and `ConnectionError` retry handling and migrated package versioning to dynamic `importlib.metadata.version("GEOS5FP")`.
- **`v1.2.4` (2025-10-24):** Added nearest-neighbor resampling when reading granules with `Point` geometries to avoid expensive global cubic interpolation.

---

### Era 3: Crash-Resilient NetCDF Validation & SSL Hardening (`v1.3.0` – `v1.3.2`, October 2025 – November 2025)

Production runs revealed two systemic failure modes: truncated NetCDF files causing C++ `SIGABRT`/segfaults inside GDAL/HDF5 (terminating the entire Python process), and SSL handshake/EOF errors on NASA NCCS servers.

- **`v1.3.0` (2025-10-27):** Built `validate_GEOS5FP_NetCDF_file.py` with comprehensive checks (byte size, filename regex pattern, CRS/bounds, variable structure, and sample NaN ratios) executed inside an isolated subprocess (`_validate_in_subprocess`) with JSON IPC and timeout protection.
- **`v1.3.1` (2025-10-28):** Added `GEOS5FPSSLError` with actionable troubleshooting guidance and automatic SSL verification fallback.
- **`v1.3.2` (2025-11-05):** Added multi-stage SSL fallback strategies including a legacy `DEFAULT@SECLEVEL=1` SSL context (`create_legacy_ssl_context.py`).

---

### Era 4: OPeNDAP Point Queries, Vectorized Tables & Computed Variables (`v2.0.0` – `v2.11.0`, November 2025 – December 2025)

Major architectural expansion to support calibration/validation (`calval`) table generation and multi-year point time-series without downloading full global `.nc4` granules.

- **`v2.0.0` (2025-11-24):** Introduced `GEOS5FP_point.py` (`query_geos5fp_point` and `query_geos5fp_point_multi`) using `xarray` and `netCDF4` against NASA's OPeNDAP service (`opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim`). Moved variable metadata into `GEOS5FP/variables.csv`, implemented vectorized spatio-temporal batching with coordinate grouping and time clustering, and established a comprehensive `pytest` suite.
- **`v2.0.1` (2025-11-25):** Suppressed `NumExpr` thread-count log messages on import.
- **`v2.1.0` (2025-11-25):** Added `targets_df` enrichment in `.query()`, modularized `cluster_times.py`, `calculate_RH.py`, `geometry_utils.py`, `get_variable_info.py`, and SSL helpers, added derived variables (`RH`, `Ta_C`, `Ea_Pa`, `SVP_Pa`, `VPD_kPa`, `Td_K`), and set default downsampling to `lanczos`.
- **`v2.3.0` (2025-11-25):** Added elapsed time and ETA progress tracking for batch and point queries (incorporating `v2.2.0` work).
- **`v2.4.0` (2025-12-10):** Added `Ca` alias, `wind_speed_mps`, and `decode_cf=False` OPeNDAP optimization to prevent remote dataset open hangs.
- **`v2.5.0` – `v2.6.0` (2025-12-16):** Added `PAR_proportion` and `NIR_proportion` albedo scale factors, usage examples, and variable documentation.
- **`v2.7.0` – `v2.8.0` (2025-12-18):** Added support for passing `pandas.Series`, `pd.DatetimeIndex`, lists, tuples, and NumPy arrays as `time_UTC`.
- **`v2.9.0` – `v2.9.1` (2025-12-19 – 2025-12-22):** Added per-step ETA logging, 120s socket timeouts, and retry backoff for OPeNDAP timeouts/504 errors.
- **`v2.10.0` (2025-12-23):** Unified variable retrieval methods under `.query()` and added single-point geometry broadcasting across time series.
- **`v2.11.0` (2025-12-30):** Implemented 7-day temporal chunking for long single-point OPeNDAP time-series queries.

---

### Era 5: Query Modularization, Pooch Downloads & GDAL Independence (`v2.12.0` – `v2.24.0`, December 2025 – November 2026)

Hardened file downloads for cloud-synced storage, eliminated GDAL HDF5 plugin dependencies, and standardized query return types.

- **`v2.12.0` (2026-01-02):** Extracted the `query()` engine into standalone `GEOS5FP/query.py`.
- **`v2.13.0` (2026-01-02):** Standardized single-variable non-raster queries to return 1D `numpy.ndarray` values.
- **`v2.14.0` – `v2.15.0` (2026-01-02 – 2026-01-14):** Added unwrapping for `rasters` point geometry wrapper objects in batch and single-time point queries.
- **`v2.16.0` (2026-01-15):** Disabled `tqdm` notebook progress bars to prevent UI freezing.
- **`v2.17.0` (2026-01-16):** Reduced batch time-clustering window to 1 day and fixed multi-dataset batch counts.
- **`v2.18.0` – `v2.19.0` (2026-01-16 – 2026-01-20):** Added `Tmin_C`, `PAR_albedo`, and `NIR_albedo`.
- **`v2.20.0` (2026-07-23):** Restored single-time raster snapshot routing in `GEOS5FPConnection.query()`.
- **`v2.21.0` (2026-10-12):** Fixed default `expected_hours` (`[1.5, 4.5, ..., 22.5]`) for 3-hourly `tavg3` products.
- **`v2.22.0` – `v2.23.0` (2026-11-10):** Migrated file downloads to `pooch.HTTPDownloader` with out-of-tree temporary staging (`pooch.os_cache("GEOS5FP")`), atomic cross-device publication (`_publish_staged_file`), and progress bars—resolving write timeouts on OneDrive-backed directories.
- **`v2.24.0` (2026-11-17):** Added pure-`netCDF4` reading (`GEOS5FPGranule._read_via_netcdf4`) and validation (`_validate_with_netcdf4`) fallbacks when GDAL's `gdal_HDF5.so` plugin is unavailable, plus inter-process file locking (`_granule_file_lock`) and `albedo_visible`/`albedo_NIR` computed variables; raised minimum Python to `>=3.11`.

---

## Part II: Pending Development Plan for `v2.25.0`

Version `2.25.0` will address all **12 open GitHub issues (`#60`–`#71`)** across four workstreams, designed to be executed and verified in a single cohesive release pass.

### Workstream 1: Query Routing, Single-Element Sequences & Multi-Variable Rasters
**Issues Covered:** `#60`, `#65`, `#66`  
**Primary File:** `GEOS5FP/query.py`

1. **Normalize Variable Sequences & Aliases (`#60`):**
   - Accept `variables: Union[str, List[str]] = None` alongside `target_variables` and `variable_name` in `query()`.
   - Validate non-empty sequences: if `len(variable_names) == 0`, raise `ValueError("At least one variable must be specified")`.
   - Set `single_variable = len(variable_names) == 1` so single-element sequences (e.g., `["CO2SC"]`) unwrap to the standard single-variable `Raster` path when queried against a `RasterGrid` or `RasterGeometry`.
2. **Multi-Variable Raster Queries (`#65`):**
   - When `len(variable_names) > 1` and `geometry` is a gridded geometry (or `None` for native global grid), iterate over `variable_names`, retrieve each variable's `Raster`, and return a `dict[str, Raster]` mapping each requested variable name to its raster.
3. **Standalone `query()` Execution (`#66`):**
   - When `query()` is called as a standalone function (`self is None`), lazily instantiate a default `GEOS5FPConnection()` so raster snapshot and computed-variable queries succeed without requiring a bound instance.

---

### Workstream 2: Complete Computed Variables, Unit Scaling & Temporal Interpolation
**Issues Covered:** `#61`, `#62`, `#63`, `#64`  
**Primary Files:** `GEOS5FP/constants.py`, `GEOS5FP/query.py`

1. **Complete `VARIABLE_TRANSFORMATIONS` & Apply Consistently (`#63`):**
   - Add `'Tmin_C': lambda x: x - 273.15`, `'vapor_gccm': lambda x: x / 10.0`, and `'ozone_cm': lambda x: x / 1000.0` to `VARIABLE_TRANSFORMATIONS` in `GEOS5FP/constants.py`.
   - Apply `VARIABLE_TRANSFORMATIONS` across all query branches in `GEOS5FP/query.py` (single-point, multi-point, time-series, and batch queries) so OPeNDAP point results match the units of gridded raster methods.
2. **Unified Computed-Variable Dependency Resolver & Evaluator (`#61`, `#62`):**
   - Implement a shared helper (`_expand_computed_dependencies` and `_evaluate_computed_variables`) supporting all **14** variables in `COMPUTED_VARIABLES`:
     - `Ta_C` = `Ta - 273.15`
     - `Tmin_C` = `Tmin - 273.15`
     - `RH` = `calculate_RH(Q, PS, Ta)`
     - `SVP_Pa` = `611.2 * exp((17.67 * (Ta - 273.15)) / ((Ta - 273.15) + 243.5))`
     - `Ea_Pa` = `RH * SVP_Pa`
     - `VPD_kPa` = `clip(SVP_Pa - Ea_Pa, 0, None) / 1000.0`
     - `Td_K` = `Ta - (100.0 - RH * 100.0) / 5.0`
     - `wind_speed_mps` = `sqrt(U2M**2 + V2M**2)`
     - `PAR_proportion` = `clip(ALBVISDR / ALBEDO, 0, 1)`
     - `NIR_proportion` = `clip(ALBNIRDR / ALBEDO, 0, 1)`
     - `PAR_albedo` = `clip(ALBVISDR + ALBVISDF, 0, 1)`
     - `NIR_albedo` = `clip(ALBNIRDR + ALBNIRDF, 0, 1)`
     - `albedo_visible` = `ALBVISDR`
     - `albedo_NIR` = `ALBNIRDR`
   - Invoke this shared resolver/evaluator in **batch**, **time-series**, and **single-time point** query modes, stripping intermediate dependency columns not explicitly requested by the caller.
3. **Linear Temporal Interpolation in Batch Queries (`#64`):**
   - In the batch query branch of `GEOS5FP/query.py`, honor `temporal_interpolation="interpolate"` (the default) by locating the bracketing OPeNDAP timestamps `t_before <= target_time <= t_after` and linearly interpolating `(1 - w) * val_before + w * val_after` (falling back to nearest neighbor at boundary edges or when `temporal_interpolation="nearest"`).

---

### Workstream 3: Network Resilience, Thread Safety & Cache Invalidation
**Issues Covered:** `#67`, `#68`, `#69`  
**Primary Files:** `GEOS5FP/GEOS5FP_point.py`, `GEOS5FP/GEOS5FP_connection.py`

1. **Thread-Safe OPeNDAP Multi-Variable Queries with Retries (`#67`):**
   - Refactor `query_geos5fp_point_multi()` in `GEOS5FP/GEOS5FP_point.py` to wrap `_query_geos5fp_point_multi_impl()` in the same retry/exponential-backoff loop used by `query_geos5fp_point()`.
   - Remove process-wide `socket.setdefaulttimeout()` mutations (or guard restoration inside `try ... finally` blocks) so concurrent threads do not clobber global socket settings.
2. **TTL Cache Expiration & Refresh Control (`#68`):**
   - Store `(timestamp, listing)` tuples in `GEOS5FPConnection._listings` with a configurable `listing_ttl_seconds` (defaulting to `300` seconds) and add `refresh: bool = False` to `list_remote_directory()`.
3. **Remove Dead `save_products` & Harden `HEAD` Requests (`#69`):**
   - Deprecate/remove unused `self.save_products` (or retain as a no-op keyword argument for backward compatibility while removing dead references).
   - Route directory and granule existence checks (`_check_remote`, `is_year_available`, `months_available_in_year`, `is_month_available`, `dates_available_in_month`, `is_day_available`, `list_GEOS5FP_granules`) through `create_robust_session()` with explicit timeouts instead of bare `requests.head()`.
   - Replace deprecated `datetime.utcnow()` with `datetime.now(timezone.utc)`.

---

### Workstream 4: Repository Hygiene, Dependency Trimming & Test Coverage
**Issues Covered:** `#70`, `#71`  
**Primary Files:** `pyproject.toml`, `makefile`, `examples/`, `tests/`, `README.md`

1. **Dependency Hygiene (`#71`):**
   - Move `ECOv002-calval-tables` and `sentinel-tiles` from `project.dependencies` to `[project.optional-dependencies]` under `examples` in `pyproject.toml`.
   - Remove unused `pytictoc` and `dask` from `project.dependencies`.
   - Bump package version to `2.25.0`.
2. **Relocate Root Scripts & Convert Root Tests (`#70`):**
   - Move `debug_COT_timeseries.py`, `reproduce_geos5fp_download_timeout.py`, `generate_BESSJPL_GEOS5FP_inputs.py`, `generate_BESSJPL_GEOS5FP_inputs_single_day.py`, `generate_BESSJPL_GEOS5FP_inputs_with_albedo_visible.py`, and `generate_FLiESANN_GEOS5FP_inputs.py` from the repository root into `examples/`.
   - Migrate `test_numpy_return.py` and `test_query_single_variable.py` into `tests/` using mocked OPeNDAP calls so `pytest` exercises them offline without live network traffic.
3. **Comprehensive Unit Test Suite & Documentation Update:**
   - Update `tests/test_variable_queries.py` to test single-element list queries on `RasterGrid`, `variables=` alias, multi-variable raster `dict` returns, all 14 computed variables across point/time-series/batch modes, `vapor_gccm`/`ozone_cm` unit conversions, batch linear interpolation, and standalone `query()`.
   - Update `tests/test_connection.py` to test `_listings` TTL expiration, `refresh=True`, and robust session `HEAD` checks.
   - Add `tests/test_point_queries.py` to verify `query_geos5fp_point_multi()` retry behavior and timeout cleanup.
   - Update `makefile` and `README.md` (updating the **Known Issues & Limitations**, **Release History**, and **Citation** sections for `v2.25.0`).

---

## Part III: Implementation & Verification Checklist for `v2.25.0`

| Step | Target Files | Issues Resolved | Verification |
| :--- | :--- | :--- | :--- |
| **1** | `GEOS5FP/constants.py` | `#63` | Unit tests verifying `Ta_C`, `Tmin_C`, `vapor_gccm`, and `ozone_cm` transformations |
| **2** | `GEOS5FP/query.py` | `#60`, `#61`, `#62`, `#63`, `#64`, `#65`, `#66` | `pytest tests/test_variable_queries.py tests/test_series_input.py` |
| **3** | `GEOS5FP/GEOS5FP_point.py` | `#67` | `pytest tests/test_point_queries.py` |
| **4** | `GEOS5FP/GEOS5FP_connection.py` | `#68`, `#69` | `pytest tests/test_connection.py tests/test_connection_download.py` |
| **5** | `examples/*`, `tests/*`, `pyproject.toml`, `makefile` | `#70`, `#71` | Clean root directory; full `pytest` suite passes offline in `<3s` |
| **6** | `README.md` | Release docs | Verify version `2.25.0`, updated Known Issues, and citation blocks |