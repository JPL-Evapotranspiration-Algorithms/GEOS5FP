# `GEOS5FP` Pending Updates & Engineering Report (`v2.25.0` $\rightarrow$ `v2.26.0`)

* **Repository:** [`JPL-Evapotranspiration-Algorithms/GEOS5FP`](https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP)
* **Current Version:** `v2.25.0`
* **Target Release:** `v2.26.0`
* **Date:** November 23, 2026
* **Affected Downstream Ecosystem:** `ECOv003-L3T-L4T-JET`, `JET3`, `BESS-JPL`, `FLiESANN`, `PM-JPL`, `AquaSEBS`

---

## 1. Executive Summary

Following the `v2.24.0` and `v2.25.0` releases of `GEOS5FP`—which introduced `pooch` out-of-tree staged downloads, direct `netCDF4` read/validation fallbacks independent of GDAL's HDF5 plugin, and unified `.query()` routing—integration testing across `ECOv003-L3T-L4T-JET` (`v1.23.0`) and `JET3` (`v3.2.0`) identified several remaining edge cases, silent spatial-query defects, API asymmetries, and containerized HPC runtime risks.

This report documents **eight concrete updates** targeted for `GEOS5FP v2.26.0` across four areas:

1. **Spatial & Tabular `.query()` Correctness (`GEOS5FP/query.py`):** Fixing a silent coordinate-broadcasting bug when pairing a `MultiPoint` geometry of length $N$ with a timestamp sequence of length $N$, adding an explicit `return_dict` option for gridded multi/single-variable queries, honoring `allow_fallback=True` on single-timestamp point queries, and preserving point indices in multi-point `time_range` queries.
2. **SI Unit & Backward-Compatible Method Parity (`GEOS5FP/GEOS5FP_connection.py` & `GEOS5FP/constants.py`):** Adding `Td_C()` and computed variable `'Td_C'` to match `Ta_C()` and `Tmin_C()`, providing a deprecated `wind_speed()` compatibility wrapper over `wind_speed_mps()`, and routing `COMPUTED_VARIABLES` inside `GEOS5FPConnection.variable()`.
3. **Container & Read-Only Filesystem Resilience (`GEOS5FP/GEOS5FP_connection.py` & `GEOS5FP/GEOS5FP_granule.py`):** Falling back to `tempfile.gettempdir()` when `pooch.os_cache("GEOS5FP")` resides on a read-only `$HOME` directory (common in Apptainer/Singularity clusters) and hardening `_read_via_netcdf4` dimension handling.
4. **Unit Test Coverage (`tests/`):** Adding deterministic, offline unit tests covering every fix above.

---

## 2. Prioritized Summary of Pending Updates

| ID | Priority | Component | Issue / Gap | Proposed Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **UPD-1** | **Critical** | `GEOS5FP/query.py` (`is_batch_query`) | Pairing a `MultiPoint` of length $N$ with a `time_UTC` sequence of length $N$ repeats `geoms[0]` for all $N$ rows. | Unpack `list(unwrapped.geoms)` when `len(unwrapped.geoms) == len(times)`. |
| **UPD-2** | **High** | `GEOS5FP/GEOS5FP_connection.py` (`download_file`) | `pooch.os_cache("GEOS5FP")` fails with `PermissionError` in Apptainer/Singularity containers with read-only `$HOME`. | Probe writability of `pooch.os_cache("GEOS5FP")` and fall back to `tempfile.gettempdir()`. |
| **UPD-3** | **High** | `GEOS5FP/query.py` (`query`) | `query(variables=["CO2SC"], geometry=RasterGrid)` returns a `Raster` (whose `.keys` is `None`), whereas 2+ variables return a `dict`. | Add `return_dict: bool = False` option to guarantee `dict[str, Raster]` returns when requested. |
| **UPD-4** | **Medium** | `GEOS5FP/query.py` (single-time point branch) | `allow_fallback=True` is ignored during single-timestamp `Point`/`MultiPoint` `.query()` calls when OPeNDAP fails. | Wrap OPeNDAP single-time point calls in `try/except` and fall back to `conn.interpolate()` when `allow_fallback=True`. |
| **UPD-5** | **Medium** | `GEOS5FP/GEOS5FP_connection.py` & `constants.py` | `Td_K()` exists, but `Td_C()` and computed `'Td_C'` are missing (unlike `Ta_C()` and `Tmin_C()`). | Implement `GEOS5FPConnection.Td_C()` and register `'Td_C'` in `COMPUTED_VARIABLES`. |
| **UPD-6** | **Medium** | `GEOS5FP/GEOS5FP_connection.py` (`variable`) | Calling `conn.variable("wind_speed_mps", ...)` or other `COMPUTED_VARIABLES` raises `KeyError`. | Route `variable_name in COMPUTED_VARIABLES` in `GEOS5FPConnection.variable()` to the corresponding method. |
| **UPD-7** | **Low** | `GEOS5FP/GEOS5FP_connection.py` | Un-updated external scripts calling `.wind_speed(...)` hit an immediate `AttributeError`. | Add a `wind_speed(...)` wrapper emitting `DeprecationWarning` and delegating to `wind_speed_mps(...)`. |
| **UPD-8** | **Low** | `GEOS5FP/query.py` (`time_range` + `MultiPoint`) | Multi-point `time_range` queries with `single_variable=True` interleave rows across points into an unlabeled 1D array. | Preserve `point_index` / `(lat, lon)` metadata or return a structured `GeoDataFrame` when `len(points) > 1`. |

---

## 3. Detailed Technical Specifications

### 3.1. UPD-1 (Critical): Unpack `MultiPoint` Geometries in Spatio-Temporal Batch `.query()`

* **Location:** `GEOS5FP/query.py` (lines 465–487)
* **Root Cause:**
  When downstream tabular processors such as `JET3.process_JET_table()` invoke `JET()` in online mode (`offline_mode=False`), `process_JET_table` constructs a single `rasters.MultiPoint` of length $N$ (`geometry = MultiPoint(coords, crs=WGS84)`) alongside a `list` of $N$ timestamps (`time_UTC`).
  Inside `GEOS5FP/query.py`:
  1. Because `time_UTC` is a `list`, `is_batch_query` is set to `True`.
  2. When normalizing `geometry` into a per-record list `geometries`, `isinstance(geometry, gpd.GeoSeries)` and `isinstance(geometry, list)` are both `False`, while `is_point_geometry(geometry)` returns `True` for `MultiPoint`.
  3. Consequently, `query()` executes `geometries = [geometry] * len(times)` and then extracts `pt_lon, pt_lat = unwrapped_geom.geoms[0].x, unwrapped_geom.geoms[0].y` for every record—silently querying the **first point's coordinates for all $N$ timestamps**.
* **Proposed Fix:**
  ```python
  if geometry is not None:
      unwrapped_input_geom = geometry.geometry if hasattr(geometry, "geometry") else geometry
      if isinstance(geometry, gpd.GeoSeries):
          geometries = geometry.tolist()
      elif isinstance(geometry, list):
          geometries = geometry
      elif isinstance(unwrapped_input_geom, MultiPoint):
          pts = list(unwrapped_input_geom.geoms)
          if len(pts) == len(times):
              geometries = pts
          elif len(pts) == 1:
              geometries = [pts[0]] * len(times)
          else:
              raise ValueError(
                  f"Number of points in MultiPoint ({len(pts)}) must match number of times ({len(times)})"
              )
      elif is_point_geometry(geometry):
          geometries = [geometry] * len(times)
      else:
          raise ValueError(
              "For batch queries with geometry, must provide GeoSeries, list, or Point/MultiPoint"
          )
  ```

---

### 3.2. UPD-2 (High): Writable Fallback for Pooch Staging Root in Read-Only Containers

* **Location:** `GEOS5FP/GEOS5FP_connection.py` (`download_file`, lines 423–430)
* **Root Cause:**
  `download_file()` stages streamed HTTP chunks inside `staging_root = Path(pooch.os_cache("GEOS5FP")).expanduser().resolve()`. On Linux, `pooch.os_cache("GEOS5FP")` defaults to `~/.cache/GEOS5FP` (e.g., `/root/.cache/GEOS5FP`). In production Apptainer/Singularity containers on JPL HPC nodes where `/root` is mounted read-only (and only `/tmp` and runtime scratch volumes are writable), `makedirs(staging_root, exist_ok=True)` raises `PermissionError`, which is caught and converted into `FailedGEOS5FPDownload`.
* **Proposed Fix:**
  Add a helper `_resolve_staging_root(download_root: Path) -> Path` that checks `pooch.os_cache("GEOS5FP")` first and falls back to `Path(tempfile.gettempdir()) / "GEOS5FP_cache"` if `pooch.os_cache` cannot be created or is not writable:
  ```python
  def _resolve_staging_root(download_root: Path) -> Path:
      candidates = [
          Path(pooch.os_cache("GEOS5FP")).expanduser(),
          Path(tempfile.gettempdir()) / "GEOS5FP_cache",
      ]
      for candidate in candidates:
          try:
              makedirs(candidate, exist_ok=True)
              resolved = candidate.resolve()
              if not resolved.is_relative_to(download_root) and os.access(resolved, os.W_OK):
                  return resolved
          except OSError:
              continue
      raise FailedGEOS5FPDownload(
          f"No writable staging directory available outside download directory: {download_root}"
      )
  ```

---

### 3.3. UPD-3 (High): Explicit `return_dict` Option on Gridded `.query()`

* **Location:** `GEOS5FP/query.py` (`query`, lines 428 and 750–772)
* **Root Cause:**
  When `geometry` is a `RasterGrid` or `RasterGeometry`, `query(variables=["CO2SC", "Ta_K"], ...)` returns `dict[str, Raster]`, whereas `query(variables=["CO2SC"], ...)` unwraps the 1-element sequence and returns a `rasters.Raster`. Because `rasters.Raster` defines `.keys = None`, callers expecting a dictionary mapping (such as `BESS_JPL/retrieve_BESS_JPL_GEOS5FP_inputs.py`, which calls `retrieved.keys()`) fail with `TypeError: 'NoneType' object is not callable`.
* **Proposed Fix:**
  Add `return_dict: bool = False` to `query()`:
  * Keep `return_dict=False` as the default so `query(["CO2SC"], time_UTC=..., geometry=grid)` continues returning a `Raster` (preserving `v2.25.0` behavior).
  * When `return_dict=True`, always return `{var_name: raster}` even when `len(variable_names) == 1`.

---

### 3.4. UPD-4 (Medium): Honor `allow_fallback=True` on Single-Timestamp Point Queries

* **Location:** `GEOS5FP/query.py` (lines 676–737)
* **Root Cause:**
  In `GEOS5FPConnection.variable()`, OPeNDAP point queries catch exceptions and fall back to `self.interpolate(...)` when `self.allow_fallback` is `True`. However, the dedicated single-timestamp point path in `GEOS5FP/query.py` calls `query_geos5fp_point(...)` directly and logs a warning on failure without checking `conn.allow_fallback`.
* **Proposed Fix:**
  When `query_geos5fp_point` raises an exception (or returns empty data) and `conn is not None and getattr(conn, "allow_fallback", False)` is `True`, fall back to `conn.interpolate(...)` (applying `VARIABLE_TRANSFORMATIONS` if applicable) so `allow_fallback=True` behaves consistently across both `.variable()` and `.query()`.

---

### 3.5. UPD-5 & UPD-6 (Medium): `Td_C` Parity & `COMPUTED_VARIABLES` Routing in `.variable()`

* **Locations:** `GEOS5FP/GEOS5FP_connection.py` and `GEOS5FP/constants.py`
* **Root Cause:**
  1. `GEOS5FPConnection` exposes `Ta_K()` + `Ta_C()` and `Tmin_K()` + `Tmin_C()`, but only `Td_K()` (without `Td_C()`), requiring downstream models to perform manual Kelvin-to-Celsius conversion (`Td_K(...) - 273.15`).
  2. Calling `conn.variable("wind_speed_mps", ...)` or `conn.variable("Ta_C", ...)` raises `KeyError` in `get_variable_info()` because `GEOS5FPConnection.variable()` only checks `GEOS5FP_VARIABLES` rather than checking `COMPUTED_VARIABLES` first.
* **Proposed Fix:**
  1. Add `Td_C()` to `GEOS5FPConnection`:
     ```python
     def Td_C(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
         return self.Td_K(time_UTC=time_UTC, geometry=geometry, resampling=resampling) - 273.15
     ```
  2. Add `'Td_C'` to `COMPUTED_VARIABLES` in `GEOS5FP/constants.py`, `COMPUTED_DEPENDENCIES` (`'Td_C': ['Q', 'PS', 'Ta']`), and `_evaluate_computed_variable` in `GEOS5FP/query.py`.
  3. In `GEOS5FPConnection.variable()`, if `variable_name in COMPUTED_VARIABLES and hasattr(self, variable_name)`, delegate to `getattr(self, variable_name)(time_UTC=time_UTC, geometry=geometry, resampling=resampling)`.

---

### 3.6. UPD-7 (Low): Deprecated `wind_speed()` Compatibility Wrapper

* **Location:** `GEOS5FP/GEOS5FP_connection.py`
* **Root Cause:**
  `GEOS5FP >=2.24.0` standardized on `wind_speed_mps(...)` and removed `wind_speed(...)`. While `JET3 v3.2.0` now uses `wind_speed_mps(...)`, external user scripts or older notebooks may still call `conn.wind_speed(...)`.
* **Proposed Fix:**
  Provide a thin compatibility wrapper emitting a `DeprecationWarning`:
  ```python
  def wind_speed(self, time_UTC: Union[datetime, str], geometry: RasterGeometry = None, resampling: str = None) -> Raster:
      warnings.warn(
          "GEOS5FPConnection.wind_speed() is deprecated; use wind_speed_mps() instead.",
          DeprecationWarning,
          stacklevel=2,
      )
      return self.wind_speed_mps(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
  ```

---

### 3.7. UPD-8 (Low): Preserve Point Identity in `MultiPoint` `time_range` Queries

* **Location:** `GEOS5FP/query.py` (lines 591–673)
* **Root Cause:**
  When `time_range=(start, end)` is queried with a `MultiPoint` of $M > 1$ locations and `single_variable=True`, `all_variable_dfs` concatenates each point's time series along `axis=0` and calls `.sort_index()`, returning a flat 1D `np.ndarray` of length $M \times T$ with timestamps interleaved across points and no indication of which element belongs to which point.
* **Proposed Fix:**
  When `len(points) > 1` in a `time_range` query, attach a `point_index` column (or return the time-indexed `GeoDataFrame` containing `geometry` when `len(points) > 1` unless explicitly reduced) so multi-point time-series outputs remain unambiguous.

---

## 4. Verification & Testing Plan for `GEOS5FP v2.26.0`

All pending updates should be verified with fast, network-free unit tests in `tests/` before cutting `v2.26.0`:

| Test Function | Target File | Verification Criteria |
| :--- | :--- | :--- |
| `test_batch_query_multipoint_unpacks_distinct_coordinates` | `tests/test_variable_queries.py` | Passing a 3-point `MultiPoint` with 3 timestamps queries 3 distinct `(lat, lon)` coordinates rather than repeating `geoms[0]`. |
| `test_batch_query_multipoint_length_mismatch_raises` | `tests/test_variable_queries.py` | Passing a 2-point `MultiPoint` with 3 timestamps raises `ValueError`. |
| `test_download_staging_falls_back_when_os_cache_readonly` | `tests/test_connection_download.py` | Mocking `pooch.os_cache` to a non-writable path redirects staging to `tempfile.gettempdir()` without raising `FailedGEOS5FPDownload`. |
| `test_query_return_dict_on_single_raster_variable` | `tests/test_variable_queries.py` | `conn.query(variables=["CO2SC"], geometry=grid, return_dict=True)` returns `{"CO2SC": sentinel}` while `return_dict=False` returns `sentinel`. |
| `test_single_point_query_allow_fallback_uses_interpolate` | `tests/test_point_queries.py` | When OPeNDAP raises `TimeoutError` and `allow_fallback=True`, `conn.query("Ta_K", time_UTC=..., geometry=Point(...))` falls back to `conn.interpolate()`. |
| `test_td_c_method_and_computed_variable` | `tests/test_variable_queries.py` | `conn.Td_C(...)` and `conn.query(["Td_C"], ...)` equal `Td_K - 273.15` across raster, point, and batch queries. |
| `test_variable_method_routes_computed_variables` | `tests/test_variable_queries.py` | `conn.variable("wind_speed_mps", ...)` and `conn.variable("Ta_C", ...)` succeed without raising `KeyError`. |
| `test_wind_speed_deprecated_alias` | `tests/test_variable_queries.py` | `conn.wind_speed(...)` emits `DeprecationWarning` and delegates to `conn.wind_speed_mps(...)`. |

---

## 5. Downstream Rollout Sequence

1. **Release `GEOS5FP v2.26.0`:** Implement `UPD-1` through `UPD-8`, run `pytest -vv`, bump `pyproject.toml` to `2.26.0`, tag `v2.26.0`, and publish to PyPI.
2. **Release `BESS-JPL` patch (if needed):** Guard `retrieve_BESS_JPL_GEOS5FP_inputs.py` against non-`dict` returns (or pass `return_dict=True` / call `GEOS5FP_connection.CO2SC(...)` directly) and publish.
3. **Finalize `ECOv003-L3T-L4T-JET v1.23.0`:**
   * Pin `GEOS5FP~=2.26.0`, `JET3~=3.2.0`, and the updated `BESS-JPL` version in `ECOv003-L3T-L4T-JET/pyproject.toml`.
   * Re-run `python examples/example_L3T_L4T_JET_run_with_download.py`, `make test`, `make smoke-test`, and `make docker-smoke-test`.