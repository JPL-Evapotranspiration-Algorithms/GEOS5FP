# Comprehensive Development Plan: `GEOS5FP` (`v1.0.1` – `v2.27.0`)

**Author:** Gregory H. Halverson (NASA Jet Propulsion Laboratory)

**Repository:** [JPL-Evapotranspiration-Algorithms/GEOS5FP](https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP)

**Date:** November 23, 2026

**Completed Milestones:** `v1.0.1` – `v2.27.0` (180 passing unit tests)

**Downstream Ecosystem:** `ECOv003-L3T-L4T-JET`, `JET3`, `BESS-JPL`, `FLiESANN`, `PM-JPL`, `AquaSEBS`

## Executive Summary

The `GEOS5FP` Python package provides programmatic retrieval, spatio-temporal interpolation, empirical downscaling, and tabular extraction for NASA's **GEOS-5 Forward Processing (GEOS-5 FP)** near-real-time meteorological assimilation products.

Across 45 releases from `v1.0.1` (July 2024) through `v2.27.0` (November 2026), `GEOS5FP` evolved from a monolithic raster download-and-interpolation script into a modular, multi-modal scientific data access engine supporting:

* **Gridded raster retrieval and empirical downscaling** via `rasters`, `netCDF4`, and GDAL

* **High-throughput OPeNDAP point, multi-point, time-series, and batch table queries** via `xarray`, `netCDF4`, `pandas`, and `geopandas`

* **Automatic evaluation and consistent physical unit scaling of derived meteorological variables** across gridded, point, time-series, and batch modes

* **Subprocess-isolated NetCDF4 validation, inter-process file locking, and out-of-tree Pooch staged downloads** resilient to C++ library crashes, missing GDAL HDF5 plugins, read-only container filesystems, and cloud-synced directories (OneDrive/Google Drive)

This document consolidates the complete release history (`v1.0.1` – `v2.27.0`), the engineering specifications for `v2.26.0` (`#74`–`#81`) and `v2.27.0` (`#83`), and the coordinated multi-repository rollout across `JET3` and `ECOv003-L3T-L4T-JET`.

## Part I: Completed Development History (`v1.0.1` – `v2.25.0`)

### Era 1: Monolithic Prototype & Initial PyPI Release (`v1.0.1` – `v1.0.7`, June 2024 – March 2025)

* **`v1.0.1` (2024-07-20):** Initial PyPI release (`geos5fp`), containing `geos5fp/geos5fp.py` and `geos5fp/downscaling/downscaling.py` implementing HTTP directory scraping (`BeautifulSoup`), `wget` downloads, linear temporal interpolation (`before_and_after` / `interpolate`), and empirical spatial downscaling (`linear_downscale`, `bias_correct`, `downscale_air_temperature`, `downscale_soil_moisture`, `downscale_vapor_pressure_deficit`, and `downscale_relative_humidity`).

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

### Era 5: Query Modularization, Pooch Staged Downloads, GDAL Independence & Query Parity (`v2.12.0` – `v2.25.0`, December 2025 – November 2026)

* **`v2.12.0` – `v2.21.0` (2026-01-02 – 2026-10-12):** Extracted `query()` into `GEOS5FP/query.py`; standardized single-variable non-raster queries to return `numpy.ndarray`; added unwrapping for `rasters` point geometry wrappers; added `Tmin_C`, `PAR_albedo`, and `NIR_albedo`; restored single-time raster snapshot routing; and fixed default `expected_hours` (`[1.5, 4.5, ..., 22.5]`) for 3-hourly `tavg3` products.

* **`v2.22.0` – `v2.23.0` (2026-11-10):** Replaced custom `requests` chunk loops and removed `GEOS5FP/download_file.py` in favor of `pooch.HTTPDownloader` with per-attempt temporary staging under `pooch.os_cache("GEOS5FP")` (outside cloud-synced `download_directory` paths), atomic cross-device publication (`_publish_staged_file`), verified HTTPS, and progress bars.

* **`v2.24.0` (2026-11-17):** Switched primary NetCDF validation to `netCDF4.Dataset` (`_validate_with_netcdf4`) and added `GEOS5FPGranule._read_via_netcdf4` fallback so valid `.nc4` granules load reliably even when GDAL's `gdal_HDF5.so` plugin is absent; fixed subprocess `ImportError` JSON serialization; added inter-process file locking (`_granule_file_lock`); registered `albedo_visible` and `albedo_NIR`; and set minimum Python support to `>=3.11`.

* **`v2.25.0` (2026-11-23):** Resolved `#60`–`#71` (unified 1-element sequences and `variables=` alias on gridded geometries; added `dict[str, Raster]` multi-variable raster returns and standalone `query()` raster support; implemented all 14 `COMPUTED_VARIABLES` and OPeNDAP unit scaling; added linear temporal interpolation in batch queries; added socket timeout locking, retry backoff on `query_geos5fp_point_multi`, and TTL cache expiration; trimmed runtime dependencies and relocated scripts to `examples/`).

## Part II: Completed Spatial Query & Container Resilience Release (`v2.26.0`, November 23, 2026)

Integration testing across `ECOv003-L3T-L4T-JET v1.23.0` and `JET3 v3.2.0` identified eight spatial-query edge cases, container filesystem constraints, and API parity items (`#74`–`#81`), all resolved in **`GEOS5FP v2.26.0`**:

1. **Spatial & Tabular `.query()` Correctness (`#74`, `#76`, `#77`, `#81` — `GEOS5FP/query.py`):**

   * **Issue `#74` (`UPD-1`, Critical) — Unpack `MultiPoint` Geometries in Spatio-Temporal Batch `.query()`:** When callers such as `JET3.process_JET_table()` passed a single `rasters.MultiPoint` of length $N$ alongside a `list` of $N$ timestamps, `is_point_geometry(geometry)` previously caused `query()` to broadcast `[geometry] * len(times)` and extract `geoms[0]` for every row. Fixed by unpacking `pts = list(unwrapped_input_geom.geoms)` when `len(pts) == len(times)` (broadcasting single-point `MultiPoint`s and raising `ValueError` on length mismatch).

   * **Issue `#76` (`UPD-3`, High) — Explicit `return_dict` Option on Gridded `.query()`:** Added `return_dict: bool = False` to `query()` so callers expecting `dict[str, Raster]` (such as `BESS_JPL`) receive a dictionary even when querying a single variable (`variables=["CO2SC"]`), while preserving `return_dict=False` returning a `Raster` by default.

   * **Issue `#77` (`UPD-4`, Medium) — Honor `allow_fallback=True` on Single-Timestamp Point Queries:** Added `_fallback_single_point_via_interpolate()` so single-timestamp `Point`/`MultiPoint` `.query()` calls fall back to `conn.interpolate()` (with `VARIABLE_TRANSFORMATIONS`) when OPeNDAP fails and `allow_fallback=True`.

   * **Issue `#81` (`UPD-8`, Low) — Preserve Point Identity in `MultiPoint` `time_range` Queries:** Attached a `point_index` column and returned a time-indexed `GeoDataFrame` containing `geometry` whenever `len(points) > 1` so multi-point time series are never interleaved into an unlabeled 1D array.

2. **Read-Only Container Staging Resilience (`#75` — `GEOS5FP/GEOS5FP_connection.py`):**

   * **Issue `#75` (`UPD-2`, High) — Writable Fallback for Pooch Staging Root:** Added `_resolve_staging_root()` to probe `pooch.os_cache("GEOS5FP")` (`~/.cache/GEOS5FP`) first and fall back to `Path(tempfile.gettempdir()) / "GEOS5FP_cache"` when `$HOME` is mounted read-only in HPC Apptainer/Singularity containers.

3. **SI Unit & Backward-Compatible Method Parity (`#78`, `#79`, `#80` — `GEOS5FP_connection.py`, `constants.py`, `query.py`):**

   * **Issue `#78` (`UPD-5`, Medium):** Implemented `GEOS5FPConnection.Td_C()` (`self.Td_K(...) - 273.15`) and registered `'Td_C'` in `COMPUTED_VARIABLES` and `COMPUTED_DEPENDENCIES`.

   * **Issue `#79` (`UPD-6`, Medium):** Routed `variable_name in COMPUTED_VARIABLES` inside `GEOS5FPConnection.variable()` to the corresponding computed method before calling `get_variable_info()`.

   * **Issue `#80` (`UPD-7`, Low):** Added a deprecated `GEOS5FPConnection.wind_speed()` wrapper emitting `DeprecationWarning` and delegating to `wind_speed_mps()`.

## Part III: Completed Gridded Raster Unit Conversion Fix (`v2.27.0`, November 23, 2026)

### 1. Root-Cause Investigation & Downstream Impact (`#83`)

During end-to-end validation of `ECOv003-L3T-L4T-JET v1.23.0` (`JET3 v3.3.0`, `GEOS5FP v2.26.0`, `FLiESANN v2.13.1`) against public ECOSTRESS Collection 3 reference granules (`orbit=44841`, `scene=001`, `tile=11SPS`, `2026-06-02T18:00:12Z`, `BuildID 0700`), `Ta`, `RH`, and `SM` matched the public reference product ($r > 0.999$), whereas incoming global shortwave radiation (`Rg` in `L3T_ETAUX`) exhibited a systematic **`+74.35 W/m²`** positive bias (`1064.17 W/m²` local vs. `989.83 W/m²` public, `RMSE = 74.757 W/m²`, $r = 0.38894$).

Systematic hypothesis testing ruled out missing `GEOS5FP.SWin` bias correction (`BIAS_CORRECT_FLIES_ANN = True` degraded `Bias` to `-237.25 W/m²`), Cloud Optical Thickness zeroing (`GEOS5FP.COT` was `0.0000` across the clear-sky overpass), and `FLiESANN` model weight changes, isolating the root cause in `GEOS5FP` (`v2.24.0`–`v2.26.0`):

* **`VARIABLE_TRANSFORMATIONS`** in `GEOS5FP/constants.py` defined `'vapor_gccm': lambda x: x / 10.0` ($\text{kg m}^{-2} \rightarrow \text{g cm}^{-2}$) and `'ozone_cm': lambda x: x / 1000.0` ($\text{DU} \rightarrow \text{cm}$ / $\text{atm-cm}$), but was applied **only** in `GEOS5FP/query.py` (OPeNDAP point/batch paths) and `GEOS5FPConnection._query_point_via_opendap()`.

* On gridded raster queries (`GEOS5FPConnection.vapor_gccm()`, `GEOS5FPConnection.ozone_cm()`, `conn.variable()`, and `conn.query(..., geometry=RasterGrid)`), `.query()` routed through `GEOS5FPConnection.variable()` $\rightarrow$ `GEOS5FPConnection.interpolate()`, returning unscaled `TQV` (`10.7180 kg/m²` instead of `1.0718 g/cm²` — **$10\times$ too high**) and unscaled `TO3` (`334.4433 DU` instead of `0.3344 cm` — **$1000\times$ too high**).

* Re-running `FLiESANN` on the `2026-06-02` scene with `vapor_gccm / 10.0` and `ozone_cm / 1000.0` reduced `Rg` **`Bias` from `+74.349 W/m²` to `-0.02331 W/m²`**, reduced **`RMSE` from `74.757 W/m²` to `0.52714 W/m²`**, and restored **Pearson $r$ from `0.38894` to `0.99859`**.

### 2. Implemented Changes & Verification (`v2.27.0` — 180 Passing Tests)

* **`GEOS5FP/GEOS5FP_connection.py`:** Applied `VARIABLE_TRANSFORMATIONS[variable_name]` in `GEOS5FPConnection.variable()` immediately after both `self.interpolate(...)` calls (gridded raster path and point fallback path) and before `rt.clip(result, clip_min, clip_max)`—ensuring transformations run once on raster/fallback queries without double-scaling OPeNDAP point/batch queries.

* **`tests/test_variable_queries.py`:** Added `test_vapor_gccm_and_ozone_cm_raster_unit_conversions` and `test_vapor_gccm_and_ozone_cm_point_query_no_double_scaling` verifying `TQV / 10.0` and `TO3 / 1000.0` across `vapor_gccm()`, `ozone_cm()`, `query(..., geometry=grid)`, and `query(..., geometry=Point(...))`, bringing the suite to **180 passing unit tests**.

* **`pyproject.toml` & `README.md`:** Bumped version to `2.27.0` and updated the citation and release log.

## Part IV: Downstream Rollout & Verification Checklist

1. **Publish `GEOS5FP v2.27.0`:**

   * Commit `GEOS5FP/GEOS5FP_connection.py`, `tests/test_variable_queries.py`, `README.md`, and `pyproject.toml`, merge into `main`, and publish GitHub release `v2.27.0` (triggering automated PyPI publication via `.github/workflows/python-publish.yml`).

2. **Release `JET3 v3.3.0`:**

   * Keep `BIAS_CORRECT_FLIES_ANN = False` as the default in `JET3/constants.py` while preserving the optional `bias_correct_FLiES_ANN: bool = BIAS_CORRECT_FLIES_ANN` keyword parameter on `JET()`.

   * Optionally emit a defensive warning in `JET()` if `np.nanmean(ozone_cm) > 5.0` or `np.nanmean(vapor_gccm) > 15.0` to catch upstream unit anomalies early.

3. **Finalize `ECOv003-L3T-L4T-JET v1.23.0`:**

   * Pin `"GEOS5FP>=2.27.0"` and `"JET3~=3.3.0"` in `ECOv003-L3T-L4T-JET/pyproject.toml`.

   * Ensure `BIAS_CORRECT_FLIES_ANN = False` (alongside `ZERO_COT_CORRECTION = True`) in `ECOv003_L3T_L4T_JET/constants.py` so solar radiation matches the Collection 3 operational baseline (`v1.11.0`–`v1.22.1`).

   * Add physical unit-range assertions (`vapor_gccm < 10.0` and `ozone_cm < 1.0`) in `ECOv003_L3T_L4T_JET/smoke_test.py` and `tests/test_docker_smoke.py`.

   * Re-run `examples/example_L3T_L4T_JET_run_with_download.py` on `orbit=44841, scene=001, tile=11SPS` (`2026-06-02T18:00:12Z`) and run `make test`, `make smoke-test`, and `make docker-smoke-test` to verify full `MATCH` across `L3T_JET`, `L3T_ETAUX`, `L4T_ESI`, and `L4T_WUE`.