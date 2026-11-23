# Consolidated Development Plan: `GEOS5FP` (`v1.0.1` – `v2.25.0+`)

**Author:** Gregory H. Halverson (NASA Jet Propulsion Laboratory)  
**Repository:** [JPL-Evapotranspiration-Algorithms/GEOS5FP](https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP)  
**Date:** November 23, 2026  
**Completed Milestones:** `v1.0.1` – `v2.25.0` (173 passing tests)  

---

## Executive Summary

The `GEOS5FP` Python package provides programmatic retrieval, spatio-temporal interpolation, thermal/moisture downscaling, and tabular extraction for NASA's **GEOS-5 Forward Processing (GEOS-5 FP)** near-real-time meteorological assimilation products.

Across 43 releases from `v1.0.1` (July 2024) through `v2.25.0` (November 2026), `GEOS5FP` evolved from a monolithic raster download-and-interpolation script into a modular, multi-modal scientific data access engine supporting:

* **Gridded raster retrieval and empirical downscaling** via `rasters`, `netCDF4`, and GDAL
* **High-throughput OPeNDAP point, multi-point, time-series, and batch table queries** via `xarray`, `netCDF4`, `pandas`, and `geopandas`
* **Automatic evaluation of all 14 derived meteorological variables** across gridded, point, time-series, and batch modes
* **Subprocess-isolated NetCDF4 validation, inter-process file locking, and out-of-tree Pooch staged downloads** resilient to C++ library crashes, missing GDAL HDF5 plugins, and cloud-synced filesystems (OneDrive/Google Drive)

---

## Part I: Completed Development History (`v1.0.1` – `v2.24.0`)

### Era 1: Monolithic Prototype & Initial PyPI Release (`v1.0.1` – `v1.0.7`, June 2024 – March 2025)

* **`v1.0.1` (2024-07-20):** Initial PyPI release (`geos5fp`), containing a 1,774-line `geos5fp/geos5fp.py` module and `geos5fp/downscaling/downscaling.py` implementing HTTP directory scraping (`BeautifulSoup`), `wget` downloads, linear temporal interpolation (`before_and_after` / `interpolate`), and empirical spatial downscaling (`linear_downscale`, `bias_correct`, `downscale_air_temperature`, `downscale_soil_moisture`, `downscale_vapor_pressure_deficit`, and `downscale_relative_humidity`).
* **`v1.0.2` – `v1.0.7` (2025-02-26 – 2025-03-10):** Modularized the package into `GEOS5FPConnection`, `GEOS5FPGranule`, `HTTP_listing`, `constants`, `exceptions`, and `Timer`; fixed relative imports; configured GitHub Actions CI; and documented `wget` installation via `brew`/`mamba`.

### Era 2: Local Cache Simplification, Uppercase Rebranding & Pure-Python Downloads (`v1.1.1` – `v1.2.4`, March 2025 – October 2025)

* **`v1.1.1` – `v1.1.2` (2025-03-28 – 2025-07-15):** Renamed the package to uppercase `GEOS5FP`; short-circuited remote HTTP checks when local files already existed; and eliminated intermediate GeoTIFF product caches (`GEOS5FP_products`) and `working_directory` in favor of a single `download_directory` (`~/data/GEOS5FP`).
* **`v1.2.0` – `v1.2.4` (2025-07-22 – 2025-10-24):** Replaced `wget` with pure-Python chunked `requests` streaming and `tqdm` progress bars; added flexible string date/time parsing (`dateutil.parser`); migrated to dynamic `importlib.metadata` versioning; added `ChunkedEncodingError` / `ConnectionError` retry handling; and added nearest-neighbor resampling for `Point` granule reads.

### Era 3: Crash-Resilient NetCDF Validation & SSL Hardening (`v1.3.0` – `v1.3.2`, October 2025 – November 2025)

* **`v1.3.0` (2025-10-27):** Introduced `validate_GEOS5FP_NetCDF_file.py` with comprehensive integrity checks (file size, filename regex, CRS/bounds, variables, NaN ratios) executed inside an isolated subprocess (`_validate_in_subprocess`) with JSON IPC and configurable timeouts to protect parent pipelines from C++ `SIGABRT`/segfault crashes in GDAL/HDF5.
* **`v1.3.1` – `v1.3.2` (2025-10-28 – 2025-11-05):** Added `GEOS5FPSSLError`, `create_robust_session`, `create_legacy_ssl_context`, and multi-stage SSL fallback handling for NASA NCCS connections.

### Era 4: OPeNDAP Point Queries, Vectorized Tables & Computed Variables (`v2.0.0` – `v2.11.0`, November 2025 – December 2025)

* **`v2.0.0` – `v2.1.0` (2025-11-24 – 2025-11-25):** Introduced `GEOS5FP_point.py` (`query_geos5fp_point`, `query_geos5fp_point_multi`) using `xarray` and `netCDF4` against NASA's OPeNDAP server (`opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim`); centralized variable mappings into `GEOS5FP/variables.csv`; implemented vectorized spatio-temporal batching (`targets_df`, coordinate grouping, `cluster_times`); added derived meteorological variables (`RH`, `Ta_C`, `Ea_Pa`, `SVP_Pa`, `VPD_kPa`, `Td_K`); set default downsampling to `lanczos`; and established a comprehensive `pytest` suite.
* **`v2.3.0` – `v2.11.0` (2025-11-25 – 2025-12-30):** Added elapsed time and ETA progress tracking; disabled xarray CF decoding/locking on remote OPeNDAP datasets; added socket timeouts and retry backoff; added `Ca` alias, `wind_speed_mps`, and albedo proportion factors; added `pandas.Series` / `DatetimeIndex` / array `time_UTC` support; and implemented 7-day temporal chunking for long single-point time-series queries.

### Era 5: Query Modularization, Pooch Staged Downloads & GDAL Independence (`v2.12.0` – `v2.24.0`, December 2025 – November 2026)

* **`v2.12.0` – `v2.21.0` (2026-01-02 – 2026-10-12):** Extracted `query()` into `GEOS5FP/query.py`; standardized single-variable non-raster queries to return `numpy.ndarray`; added unwrapping for `rasters` point geometry wrappers; added `Tmin_C`, `PAR_albedo`, and `NIR_albedo`; restored single-time raster snapshot routing; and fixed default `expected_hours` (`[1.5, 4.5, ..., 22.5]`) for 3-hourly `tavg3` products.
* **`v2.22.0` – `v2.23.0` (2026-11-10):** Replaced custom `requests` chunk loops and removed `GEOS5FP/download_file.py` in favor of `pooch.HTTPDownloader` with per-attempt temporary staging under `pooch.os_cache("GEOS5FP")` (outside cloud-synced `download_directory` paths), atomic cross-device publication (`_publish_staged_file`), verified HTTPS, and progress bars.
* **`v2.24.0` (2026-11-17):** Switched primary NetCDF validation to `netCDF4.Dataset` (`_validate_with_netcdf4`) and added `GEOS5FPGranule._read_via_netcdf4` fallback so valid `.nc4` granules load reliably even when GDAL's `gdal_HDF5.so` plugin is absent; fixed subprocess `ImportError` JSON serialization; added inter-process file locking (`_granule_file_lock`); registered `albedo_visible` and `albedo_NIR`; and set minimum Python support to `>=3.11`.

---

## Part II: Completed Stabilization & Feature-Parity Release (`v2.25.0`, November 23, 2026)

Version `2.25.0` resolved all **12 tracked GitHub issues (`#60`–`#71`)** across four coordinated workstreams, verified with **173 passing unit tests** on Python 3.14:

### Workstream 1: Query Routing, Single-Element Sequences & Multi-Variable Rasters (`#60`, `#65`, `#66`)

* **Single-Element Sequence & `variables=` Alias Normalization (`#60`):** Accepted `variables` as an alias for `target_variables`/`variable_name` in `GEOS5FP/query.py`, raised `ValueError("At least one variable must be specified")` on empty sequences, and set `single_variable = len(variable_names) == 1` so single-element sequences (e.g., `["CO2SC"]`) unwrap to the single-variable `Raster` path on `RasterGrid` geometries (unblocking `BESS-JPL`, `JET3`, and `ECOv003-L3T-L4T-JET`).
* **Multi-Variable Raster Queries (`#65`):** Returned `dict[str, Raster]` mapping variable names to interpolated rasters when `len(variable_names) > 1` on gridded geometries.
* **Standalone `query()` Support (`#66`):** Instantiated a default `GEOS5FPConnection()` when `query()` is invoked standalone (`self is None`).

### Workstream 2: Complete Computed Variables, Unit Scaling & Temporal Interpolation (`#61`, `#62`, `#63`, `#64`)

* **Unit Transformations Across OPeNDAP Modes (`#63`):** Added `Tmin_C`, `vapor_gccm` (`/ 10.0`), and `ozone_cm` (`/ 1000.0`) to `VARIABLE_TRANSFORMATIONS` in `GEOS5FP/constants.py` and applied them across batch, time-series, and single-point OPeNDAP paths.
* **Unified Computed-Variable Resolver & Evaluator (`#61`, `#62`):** Implemented `_expand_requested_variables`, `_evaluate_computed_variable`, and `_apply_computed_columns` in `GEOS5FP/query.py` to support all **14** `COMPUTED_VARIABLES` (`RH`, `Ta_C`, `Tmin_C`, `SVP_Pa`, `Ea_Pa`, `VPD_kPa`, `Td_K`, `wind_speed_mps`, `PAR_proportion`, `NIR_proportion`, `PAR_albedo`, `NIR_albedo`, `albedo_visible`, `albedo_NIR`) across batch, time-series, and single-time point queries.
* **Linear Temporal Interpolation in Batch & Single-Point Queries (`#64`):** Implemented `_interpolate_series_at_time` to honor `temporal_interpolation="interpolate"` via linear interpolation between bracketing timestamps while preserving `temporal_interpolation="nearest"`.

### Workstream 3: Network Resilience, Thread Safety & Cache Invalidation (`#67`, `#68`, `#69`)

* **Thread-Safe Socket Timeouts & Multi-Variable Retries (`#67`):** Guarded `socket.setdefaulttimeout()` with a `threading.Lock` and `try...finally` restoration (`_run_with_socket_timeout`) and wrapped `query_geos5fp_point_multi()` in `_retry_opendap_call` with exponential backoff.
* **TTL Cache Expiration & Refresh Control (`#68`):** Stored `(timestamp, listing)` entries in `GEOS5FPConnection._listings` with configurable `listing_ttl_seconds` and added `refresh: bool = False` to `list_remote_directory()` and `list_GEOS5FP_granules()`.
* **Deprecated `save_products` & Hardened `HEAD` Checks (`#69`):** Deprecated `save_products` with a `DeprecationWarning` and routed remote `HEAD` status checks through `create_robust_session()` with explicit timeouts.

### Workstream 4: Repository Hygiene, Dependency Trimming & Test Coverage (`#70`, `#71`)

* **Dependency Hygiene (`#71`):** Moved `ECOv002-calval-tables` and `sentinel-tiles` into `[project.optional-dependencies] examples` in `pyproject.toml`, removed unused `pytictoc` and `dask` runtime dependencies, and bumped package version to `2.25.0`.
* **Root Script Relocation & Offline Unit Tests (`#70`):** Relocated root debug/generation scripts into `examples/`, migrated root point-query test scripts into `tests/test_point_queries.py` using mocked OPeNDAP responses, standardized `makefile` targets, and updated `README.md` with copyable software citations and current release notes.

---

## Part III: Backlog & Future Enhancements (`v2.26.0+`)

Remaining low-priority cleanup and enhancement candidates from the repository audit to consider for future releases:

1. **Lock File Cleanup & Edge-Case Hardening:**
   * Unlink stray `<granule>.nc4.lock` files upon release in `_granule_file_lock` (or place lock files in a dedicated lock subdirectory) so `~/data/GEOS5FP` contains only `.nc4` granules.
   * Normalize `[0, 360)` longitudes and update 1-pixel fallback grid spacing (`0.3125° × 0.25°`) in `GEOS5FPGranule._read_via_netcdf4`.
   * Expand all points in a `MultiPoint` geometry when passed inside a batch `GeoSeries` rather than indexing `geoms[0]`.
2. **Dead Module & Deprecation Cleanup:**
   * Remove unused SSL helper modules (`GEOS5FP/create_legacy_ssl_context.py` and `GEOS5FP/make_head_request_with_ssl_fallback.py`).
   * Replace remaining `datetime.utcnow()` calls in `GEOS5FPConnection.latest_date_available` with `datetime.now(timezone.utc).date()` and raise a descriptive exception if `latest_time_available` exhausts retries.
3. **Additional Validation Tests & Gridded Time Series (`pooch` & `netCDF4`):**
   * Add unit tests verifying reuse of valid existing granules on disk and automatic cleanup/retry when an existing or newly downloaded granule fails `netCDF4` validation.
   * Support time-series (`time_range`) queries for gridded geometries returning a time-indexed `dict[datetime, Raster]` or 3D `xarray.DataArray`.