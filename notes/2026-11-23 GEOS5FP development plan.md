# Comprehensive Development Plan: `GEOS5FP` (`v1.0.1` – `v2.26.0`)

**Author:** Gregory H. Halverson (NASA Jet Propulsion Laboratory)  
**Repository:** [JPL-Evapotranspiration-Algorithms/GEOS5FP](https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP)  
**Date:** November 23, 2026  
**Completed Milestones:** `v1.0.1` – `v2.25.0` (173 passing unit tests)  
**Target Release:** `v2.26.0` (resolving GitHub Issues `#74`–`#81`)  

---

## Executive Summary

The `GEOS5FP` Python package provides programmatic retrieval, spatio-temporal interpolation, empirical downscaling, and tabular extraction for NASA's **GEOS-5 Forward Processing (GEOS-5 FP)** near-real-time meteorological assimilation products.

Across 43 releases from `v1.0.1` (July 2024) through `v2.25.0` (November 2026), `GEOS5FP` evolved from a monolithic raster download-and-interpolation script into a modular, multi-modal scientific data access engine supporting:

* **Gridded raster retrieval and empirical downscaling** via `rasters`, `netCDF4`, and GDAL
* **High-throughput OPeNDAP point, multi-point, time-series, and batch table queries** via `xarray`, `netCDF4`, `pandas`, and `geopandas`
* **Automatic evaluation of derived meteorological variables** across gridded, point, time-series, and batch modes
* **Subprocess-isolated NetCDF4 validation, inter-process file locking, and out-of-tree Pooch staged downloads** resilient to C++ library crashes, missing GDAL HDF5 plugins, and cloud-synced filesystems (OneDrive/Google Drive)

Integration testing across downstream repositories (`ECOv003-L3T-L4T-JET v1.23.0` and `JET3 v3.2.0`) identified eight remaining spatial-query edge cases, container filesystem constraints, and API parity items (`#74`–`#81`), which form the target scope for **`v2.26.0`**.

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

1. **Query Routing, Single-Element Sequences & Multi-Variable Rasters (`#60`, `#65`, `#66`):** Accepted `variables` as an alias, unwrapped 1-element sequences (`len(variable_names) == 1`) on gridded geometries to return `Raster` objects, returned `dict[str, Raster]` for multi-variable raster queries, and enabled standalone `query()` raster calls.
2. **Complete Computed Variables, Unit Scaling & Temporal Interpolation (`#61`, `#62`, `#63`, `#64`):** Added `Tmin_C`, `vapor_gccm` (`/ 10.0`), and `ozone_cm` (`/ 1000.0`) transformations across all OPeNDAP query modes; implemented all 14 `COMPUTED_VARIABLES` across batch, time-series, and single-time point modes; and added linear temporal interpolation (`_interpolate_series_at_time`) in batch and single-point queries.
3. **Network Resilience, Thread Safety & Cache Invalidation (`#67`, `#68`, `#69`):** Guarded `socket.setdefaulttimeout()` with a `threading.Lock` and `try...finally` restoration, added retry/backoff to `query_geos5fp_point_multi()`, added TTL expiration and `refresh=True` invalidation to `GEOS5FPConnection._listings`, deprecated `save_products`, and routed `HEAD` checks through `create_robust_session()`.
4. **Repository Hygiene, Dependency Trimming & Test Coverage (`#70`, `#71`):** Moved `ECOv002-calval-tables` and `sentinel-tiles` to `[project.optional-dependencies] examples`, removed unused `pytictoc` and `dask` runtime dependencies, relocated root scripts to `examples/`, converted root point tests to mocked unit tests, and updated `README.md` and `makefile`.

---

## Part III: Target Development Plan for `v2.26.0` (Issues `#74`–`#81`)

Integration testing across `ECOv003-L3T-L4T-JET v1.23.0` and `JET3 v3.2.0` identified **8 concrete updates (`#74`–`#81`)** to resolve in `GEOS5FP v2.26.0` across four workstreams.

### Workstream 1: Spatial & Tabular `.query()` Correctness (`#74`, `#76`, `#77`, `#81`)
**Primary File:** `GEOS5FP/query.py`

1. **Issue [#74](https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP/issues/74) (Critical) — Unpack `MultiPoint` Geometries in Spatio-Temporal Batch `.query()`:**
   * **Root Cause:** When `JET3.process_JET_table()` runs in online mode (`offline_mode=False`), it passes a single `rasters.MultiPoint` of length $N$ (`geometry = MultiPoint(coords, crs=WGS84)`) alongside a `list` of $N$ timestamps (`time_UTC`). Because `is_point_geometry(geometry)` is `True`, `query()` broadcasts `[geometry] * len(times)` and extracts `unwrapped_geom.geoms[0]` for every row—silently querying the first point's coordinates for all $N$ timestamps.
   * **Fix:** When `unwrapped_input_geom` is a `MultiPoint`, unpack `pts = list(unwrapped_input_geom.geoms)`:
     * If `len(pts) == len(times)`: set `geometries = pts`.
     * Else if `len(pts) == 1`: broadcast `geometries = [pts[0]] * len(times)`.
     * Else: raise `ValueError(f"Number of points in MultiPoint ({len(pts)}) must match number of times ({len(times)})")`.

2. **Issue [#76](https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP/issues/76) (High) — Explicit `return_dict` Option on Gridded `.query()`:**
   * **Root Cause:** On gridded geometries (`RasterGrid` / `RasterGeometry`), `query(variables=["CO2SC", "Ta_K"], ...)` returns `dict[str, Raster]`, whereas `query(variables=["CO2SC"], ...)` returns a `rasters.Raster` (whose `.keys` attribute is `None`), causing callers that call `.keys()` on the result to fail with `TypeError: 'NoneType' object is not callable`.
   * **Fix:** Add `return_dict: bool = False` to `query()`. Keep `return_dict=False` as the default so 1-element sequences return a `Raster`, and return `{var_name: raster}` when `return_dict=True` regardless of variable count.

3. **Issue [#77](https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP/issues/77) (Medium) — Honor `allow_fallback=True` on Single-Timestamp Point Queries:**
   * **Root Cause:** `GEOS5FPConnection.variable()` catches OPeNDAP failures and falls back to `self.interpolate(...)` when `self.allow_fallback` is `True`, whereas the single-timestamp point branch in `GEOS5FP/query.py` logs a warning and returns empty results without checking `conn.allow_fallback`.
   * **Fix:** When `query_geos5fp_point` raises an exception or returns empty data and `conn is not None and getattr(conn, "allow_fallback", False)` is `True`, fall back to `conn.interpolate(...)` and apply `VARIABLE_TRANSFORMATIONS`.

4. **Issue [#81](https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP/issues/81) (Low) — Preserve Point Identity in `MultiPoint` `time_range` Queries:**
   * **Root Cause:** When `time_range=(start, end)` is queried with a `MultiPoint` of $M > 1$ locations and `single_variable=True`, `all_variable_dfs` concatenates each point's series and sorts by timestamp—returning an unlabeled 1D `np.ndarray` of length $M \times T$ with rows interleaved across points.
   * **Fix:** Attach a `point_index` column and return the time-indexed `GeoDataFrame` containing `geometry` whenever `len(points) > 1` so multi-point time-series results remain unambiguous.

---

### Workstream 2: Container & Read-Only Filesystem Resilience (`#75`)
**Primary File:** `GEOS5FP/GEOS5FP_connection.py`

5. **Issue [#75](https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP/issues/75) (High) — Writable Fallback for Pooch Staging Root in Read-Only Containers:**
   * **Root Cause:** `download_file()` stages streamed HTTP chunks inside `pooch.os_cache("GEOS5FP")` (`~/.cache/GEOS5FP`), which raises `PermissionError` (mapped to `FailedGEOS5FPDownload`) in Apptainer/Singularity containers on HPC nodes where `$HOME` (`/root`) is mounted read-only.
   * **Fix:** Add `_resolve_staging_root(download_root: Path) -> Path` to probe `pooch.os_cache("GEOS5FP")` first and fall back to `Path(tempfile.gettempdir()) / "GEOS5FP_cache"` when `pooch.os_cache` cannot be created or is not writable.

---

### Workstream 3: SI Unit & Backward-Compatible Method Parity (`#78`, `#79`, `#80`)
**Primary Files:** `GEOS5FP/GEOS5FP_connection.py`, `GEOS5FP/constants.py`, `GEOS5FP/query.py`

6. **Issue [#78](https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP/issues/78) (Medium) — Add `Td_C()` Method and Register `Td_C` in `COMPUTED_VARIABLES`:**
   * **Fix:** Implement `GEOS5FPConnection.Td_C(self, time_UTC, geometry=None, resampling=None) -> Raster` (`self.Td_K(...) - 273.15`), register `'Td_C'` in `COMPUTED_VARIABLES` (`GEOS5FP/constants.py`) and `COMPUTED_DEPENDENCIES` (`'Td_C': ['Q', 'PS', 'Ta']`), and evaluate `'Td_C'` in `_evaluate_computed_variable` (`GEOS5FP/query.py`).

7. **Issue [#79](https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP/issues/79) (Medium) — Route `COMPUTED_VARIABLES` Inside `GEOS5FPConnection.variable()`:**
   * **Fix:** Check `if variable_name in COMPUTED_VARIABLES and hasattr(self, variable_name):` at the start of `GEOS5FPConnection.variable()` and delegate to `getattr(self, variable_name)(time_UTC=time_UTC, geometry=geometry, resampling=resampling)` instead of raising `KeyError`.

8. **Issue [#80](https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP/issues/80) (Low) — Deprecated `wind_speed()` Compatibility Wrapper:**
   * **Fix:** Add `GEOS5FPConnection.wind_speed(...)` emitting a `DeprecationWarning` and delegating to `self.wind_speed_mps(...)` so external scripts/notebooks using the pre-`v2.24.0` method name continue working cleanly.

---

### Workstream 4: Unit Testing & Downstream Rollout (`v2.26.0`)

| Test Function | Target File | Issues Covered | Verification Criteria |
| :--- | :--- | :--- | :--- |
| `test_batch_query_multipoint_unpacks_distinct_coordinates` | `tests/test_variable_queries.py` | `#74` | Passing a 3-point `MultiPoint` with 3 timestamps queries 3 distinct `(lat, lon)` coordinates rather than repeating `geoms[0]`. |
| `test_batch_query_multipoint_length_mismatch_raises` | `tests/test_variable_queries.py` | `#74` | Passing a 2-point `MultiPoint` with 3 timestamps raises `ValueError`. |
| `test_download_staging_falls_back_when_os_cache_readonly` | `tests/test_connection_download.py` | `#75` | Mocking `pooch.os_cache` to a non-writable path redirects staging to `tempfile.gettempdir()` without raising `FailedGEOS5FPDownload`. |
| `test_query_return_dict_on_single_raster_variable` | `tests/test_variable_queries.py` | `#76` | `conn.query(variables=["CO2SC"], geometry=grid, return_dict=True)` returns `{"CO2SC": sentinel}` while `return_dict=False` returns `sentinel`. |
| `test_single_point_query_allow_fallback_uses_interpolate` | `tests/test_point_queries.py` | `#77` | When OPeNDAP raises `TimeoutError` and `allow_fallback=True`, `conn.query("Ta_K", time_UTC=..., geometry=Point(...))` falls back to `conn.interpolate()`. |
| `test_td_c_method_and_computed_variable` | `tests/test_variable_queries.py` | `#78` | `conn.Td_C(...)` and `conn.query(["Td_C"], ...)` equal `Td_K - 273.15` across raster, point, and batch queries. |
| `test_variable_method_routes_computed_variables` | `tests/test_variable_queries.py` | `#79` | `conn.variable("wind_speed_mps", ...)` and `conn.variable("Ta_C", ...)` succeed without raising `KeyError`. |
| `test_wind_speed_deprecated_alias` | `tests/test_variable_queries.py` | `#80` | `conn.wind_speed(...)` emits `DeprecationWarning` and delegates to `conn.wind_speed_mps(...)`. |
| `test_multipoint_time_range_preserves_point_index` | `tests/test_point_queries.py` | `#81` | `time_range` query with a 2-point `MultiPoint` returns a `GeoDataFrame` with `point_index` distinguishing each point's series. |

### Downstream Rollout Sequence

1. **Release `GEOS5FP v2.26.0`:** Implement `#74`–`#81`, run `pytest -vv`, bump `pyproject.toml` to `2.26.0`, update `README.md`, tag `v2.26.0`, and publish to PyPI.
2. **Release `BESS-JPL` patch (if needed):** Guard `retrieve_BESS_JPL_GEOS5FP_inputs.py` or pass `return_dict=True` when calling `GEOS5FP_connection.query()`.
3. **Finalize `ECOv003-L3T-L4T-JET v1.23.0`:** Pin `GEOS5FP~=2.26.0` and `JET3~=3.2.0` in `ECOv003-L3T-L4T-JET/pyproject.toml` and verify with `make test`, `make smoke-test`, and `make docker-smoke-test`.