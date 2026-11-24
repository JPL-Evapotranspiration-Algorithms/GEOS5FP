# Comprehensive Development Plan: `GEOS5FP` (`v1.0.1` – `v2.27.0`)

**Author:** Gregory H. Halverson (NASA Jet Propulsion Laboratory)  
**Repository:** [JPL-Evapotranspiration-Algorithms/GEOS5FP](https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP)  
**Date:** November 23, 2026  
**Completed Milestones:** `v1.0.1` – `v2.26.0` (182 passing unit tests)  
**Target Release:** `v2.27.0` (resolving GitHub Issue `#83` and `fixing_coordinate_broadcasting` branch items)  

---

## Executive Summary

The `GEOS5FP` Python package provides programmatic retrieval, spatio-temporal interpolation, empirical downscaling, and tabular extraction for NASA's **GEOS-5 Forward Processing (GEOS-5 FP)** near-real-time meteorological assimilation products.

Across 44 releases from `v1.0.1` (July 2024) through `v2.26.0` (November 2026), `GEOS5FP` evolved from a monolithic raster download-and-interpolation script into a modular, multi-modal scientific data access engine supporting:

* **Gridded raster retrieval and empirical downscaling** via `rasters`, `netCDF4`, and GDAL
* **High-throughput OPeNDAP point, multi-point, time-series, and batch table queries** via `xarray`, `netCDF4`, `pandas`, and `geopandas`
* **Automatic evaluation of derived meteorological variables** across gridded, point, time-series, and batch modes
* **Subprocess-isolated NetCDF4 validation, inter-process file locking, and out-of-tree Pooch staged downloads** resilient to C++ library crashes, missing GDAL HDF5 plugins, read-only container filesystems, and cloud-synced directories (OneDrive/Google Drive)

End-to-end Collection 3 granule validation in `ECOv003-L3T-L4T-JET v1.23.0` (`JET3 v3.3.0`, `FLiESANN v2.13.1`) uncovered a critical unit-conversion regression on gridded raster queries (`#83`), which forms the primary target scope for **`v2.27.0`**.

---

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

---

## Part II: Completed Spatial Query & Container Resilience Release (`v2.26.0`, November 23, 2026)

Version `2.26.0` resolved all **8 tracked GitHub issues (`#74`–`#81`)** across four workstreams, verified with **182 passing unit tests** on Python 3.14:

1. **Spatial & Tabular `.query()` Correctness (`#74`, `#76`, `#77`, `#81`):**
   * Unpacked `MultiPoint` geometries in spatio-temporal batch `.query()` (`pts = list(unwrapped_input_geom.geoms)`) and validated length parity against `time_UTC` (`#74`).
   * Added explicit `return_dict: bool = False` option to `.query()` for gridded geometries so callers can reliably request `dict[str, Raster]` regardless of variable count (`#76`).
   * Honored `allow_fallback=True` on single-timestamp point `.query()` calls via `_fallback_single_point_via_interpolate()` (`#77`).
   * Preserved `point_index` and returned a `GeoDataFrame` in `MultiPoint` `time_range` queries (`#81`).
2. **Read-Only Container Staging Resilience (`#75`):**
   * Added `_resolve_staging_root()` in `GEOS5FP_connection.py` to fall back from `pooch.os_cache("GEOS5FP")` to `Path(tempfile.gettempdir()) / "GEOS5FP_cache"` when `$HOME` is mounted read-only in HPC/Apptainer containers (`#75`).
3. **SI Unit & Backward-Compatible Method Parity (`#78`, `#79`, `#80`):**
   * Added `GEOS5FPConnection.Td_C()` and registered `'Td_C'` in `COMPUTED_VARIABLES` and `COMPUTED_DEPENDENCIES` (`#78`).
   * Routed `COMPUTED_VARIABLES` inside `GEOS5FPConnection.variable()` before calling `get_variable_info()` (`#79`).
   * Added deprecated `GEOS5FPConnection.wind_speed()` wrapper delegating to `wind_speed_mps()` (`#80`).

---

## Part III: Target Development Plan for `v2.27.0` (Issue `#83` & Branch `fixing_coordinate_broadcasting`)

### Workstream 1: Gridded Raster Unit Transformation Fix (`#83` — Critical)
**Primary Files:** `GEOS5FP/GEOS5FP_connection.py`, `GEOS5FP/constants.py`, `GEOS5FP/query.py`

* **Issue [#83](https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP/issues/83) — Restore Unit Conversions for `vapor_gccm` and `ozone_cm` on Gridded Raster Queries:**
  * **Root Cause:** `VARIABLE_TRANSFORMATIONS` in `GEOS5FP/constants.py` defines `'vapor_gccm': lambda x: x / 10.0` ($\text{kg m}^{-2} \rightarrow \text{g cm}^{-2}$) and `'ozone_cm': lambda x: x / 1000.0` ($\text{DU} \rightarrow \text{cm}$ / $\text{atm-cm}$). However, `VARIABLE_TRANSFORMATIONS` is currently applied only in `GEOS5FP/query.py` (OPeNDAP point/batch paths) and `GEOS5FPConnection._query_point_via_opendap()`. When `GEOS5FPConnection.vapor_gccm()` or `GEOS5FPConnection.ozone_cm()` is called with a gridded `RasterGeometry` (or `geometry=None`), `.query()` routes through `GEOS5FPConnection.variable()` $\rightarrow$ `GEOS5FPConnection.interpolate()`, returning raw `TQV` ($10\times$ too high) and raw `TO3` ($1000\times$ too high).
  * **Downstream Impact:** Verified on ECOSTRESS Collection 3 granule `orbit=44841, scene=001, tile=11SPS` (`2026-06-02T18:00:12Z`): unscaled `vapor_gccm` (`10.7180` vs. `1.0718 g/cm²`) and `ozone_cm` (`334.4433` vs. `0.3344 cm`) shifted `FLiESANN` `SWin_Wm2` (`Rg` in `L3T_ETAUX`) by `+74.35 W/m²` (`RMSE = 74.757 W/m²`, $r = 0.38894$); applying `/ 10.0` and `/ 1000.0` restores agreement with public Collection 3 (`RMSE = 0.527 W/m²`, `Bias = -0.023 W/m²`, $r = 0.99859$).
  * **Fix:**
    1. Apply `VARIABLE_TRANSFORMATIONS[variable_name]` in `GEOS5FPConnection.variable()` for raster / `interpolate()` results (immediately after `self.interpolate(...)` and before `clip_min` / `clip_max` clipping), ensuring transformations run exactly once on raster queries without double-applying on `_query_point_via_opendap()` or `query.py` OPeNDAP paths.
    2. Audit all SI-suffixed methods (`Ta_C`, `Tmin_C`, `Td_C`, `Td_K`, `Ta_K`, `Ts_K`, `Tmin_K`, `wind_speed_mps`, `vapor_gccm`, `vapor_kgsqm`, `ozone_cm`, `ozone_dobson`, `SWin`, `SWTDN`, `AOT`, `COT`, `SM`, `PS`, `Q`, `RH`, `VPD_kPa`, `Ea_Pa`, `SVP_Pa`, `SVP_kPa`, and albedo variables) to guarantee identical physical units across raster, point, time-series, and batch queries.

---

### Workstream 2: Coordinate Broadcasting Hardening (`fixing_coordinate_broadcasting` Branch)
**Primary Files:** `GEOS5FP/query.py`, `GEOS5FP/GEOS5FP_connection.py`

* **Audit & Finalize Spatio-Temporal Broadcasting Edge Cases:**
  * Verify `MultiPoint`, `GeoSeries`, `list` of `Point`s, scalar `Point`, and `lat`/`lon` sequence broadcasting across both `GEOS5FPConnection.query()` and `GEOS5FPConnection.variable()` when `time_UTC` is passed as a `list`, `tuple`, `np.ndarray`, `pd.Series`, or `pd.DatetimeIndex`.
  * Ensure zero untracked helper scripts remain in the working tree before tagging `v2.27.0`.

---

### Workstream 3: Unit Regression Tests & Downstream Rollout (`v2.27.0`)

| Test Function | Target File | Issues Covered | Verification Criteria |
| :--- | :--- | :--- | :--- |
| `test_vapor_gccm_and_ozone_cm_raster_unit_conversions` | `tests/test_variable_queries.py` | `#83` | Mocking `conn.interpolate()` to return raw `TQV = 10.718` and `TO3 = 334.443`, verify `conn.vapor_gccm(geometry=grid)` returns `1.0718` ($\text{g cm}^{-2}$) and `conn.ozone_cm(geometry=grid)` returns `0.3344433` ($\text{cm}$). |
| `test_vapor_gccm_and_ozone_cm_query_raster_and_point_parity` | `tests/test_variable_queries.py` | `#83` | Verify `conn.query(["vapor_gccm", "ozone_cm"], geometry=grid)` and `conn.query(["vapor_gccm", "ozone_cm"], geometry=Point(...))` return consistent physical values (`TQV / 10.0` and `TO3 / 1000.0`) without double-scaling. |
| `test_all_si_variables_unit_consistency` | `tests/test_variable_queries.py` | `#83` | Verify unit-converted variables (`Ta_C`, `Tmin_C`, `Td_C`, `vapor_gccm`, `ozone_cm`) produce matching physical ranges across `.variable()`, `.query()` raster, and `.query()` point modes. |

### Downstream Rollout Sequence

1. **Release `GEOS5FP v2.27.0`:** Implement `#83` and any remaining `fixing_coordinate_broadcasting` tests, run `pytest -vv`, bump `pyproject.toml` to `2.27.0`, update `README.md`, tag `v2.27.0`, and publish to PyPI.
2. **Release `JET3 v3.3.0`:** Keep `BIAS_CORRECT_FLIES_ANN = False` as the default in `JET3/constants.py` (while retaining the `bias_correct_FLiES_ANN` keyword parameter on `JET()`), and optionally emit a defensive warning if `np.nanmean(ozone_cm) > 5.0` or `np.nanmean(vapor_gccm) > 15.0`.
3. **Update `ECOv003-L3T-L4T-JET v1.23.0`:**
   * Pin `"GEOS5FP>=2.27.0"` in `ECOv003-L3T-L4T-JET/pyproject.toml`.
   * Ensure `BIAS_CORRECT_FLIES_ANN = False` in `ECOv003_L3T_L4T_JET/constants.py`.
   * Add physical unit-range assertions (`vapor_gccm < 10.0` and `ozone_cm < 1.0`) in `ECOv003_L3T_L4T_JET/smoke_test.py` and `tests/test_docker_smoke.py`.
   * Re-run `examples/example_L3T_L4T_JET_run_with_download.py` on `orbit=44841, scene=001, tile=11SPS` (`2026-06-02T18:00:12Z`) to confirm `MATCH` across `L3T_JET`, `L3T_ETAUX`, `L4T_ESI`, and `L4T_WUE`.