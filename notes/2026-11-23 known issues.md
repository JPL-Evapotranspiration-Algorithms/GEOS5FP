# Known Issues in the `GEOS5FP` Repository

**Package:** `GEOS5FP` (`v2.24.0`)  
**Audit Date:** November 23, 2026  

This document catalogs all known bugs, API inconsistencies, performance bottlenecks, and technical debt identified in the current `GEOS5FP` (`v2.24.0`) codebase.

---

## 1. Critical / Active Functional Bugs

### 1.1. `query()` Rejects Single-Element Variable Sequences on Gridded Geometries (and Lacks `variables=` Alias)
* **Location:** `GEOS5FP/query.py` (`query()`, lines 40–81 and 378)
* **Impact:** Breaks downstream callers (`BESS-JPL`, `JET3`, `ECOv003-L3T-L4T-JET`) that dynamically build a list of missing variables (e.g., `variables=["CO2SC"]`) and pass it to `GEOS5FPConnection.query()`.
* **Details:**
  1. `query()` sets `single_variable = isinstance(target_variables, str)` rather than checking `len(variable_names) == 1`. Passing a 1-element list like `["CO2SC"]` with a `RasterGrid` hits `if not single_variable and not is_point_geometry(geometry): raise ValueError("Multiple variables can only be queried for point geometries")`.
  2. Passing an empty sequence (`[]`) falls through with `single_variable = False` instead of raising an explicit `ValueError("At least one variable must be specified")`.
  3. Downstream callers invoke `conn.query(variables=[...], ...)`, whereas `query()` only accepts `target_variables` and `variable_name`—causing `variables=` to be swallowed by `**kwargs` and raising `ValueError: target_variables parameter is required`.

### 1.2. Missing `computed_dependencies` and Formulas for Several `COMPUTED_VARIABLES` in Batch `query()`
* **Location:** `GEOS5FP/query.py` (lines 186–194, 304–342) and `GEOS5FP/constants.py` (lines 64–79)
* **Impact:** Querying several advertised `COMPUTED_VARIABLES` in batch/`targets_df` mode silently produces columns filled with `None` or drops them without computing values.
* **Details:**
  * `COMPUTED_VARIABLES` declares 14 variables: `RH`, `Ta_C`, `Tmin_C`, `Ea_Pa`, `SVP_Pa`, `VPD_kPa`, `Td_K`, `wind_speed_mps`, `PAR_proportion`, `NIR_proportion`, `PAR_albedo`, `NIR_albedo`, `albedo_visible`, and `albedo_NIR`.
  * However, `computed_dependencies` and the batch computation block in `GEOS5FP/query.py` **only implement 6 of them** (`RH`, `Ta_C`, `Tmin_C`, `wind_speed_mps`, `albedo_visible`, `albedo_NIR`).
  * Querying `Ea_Pa`, `SVP_Pa`, `VPD_kPa`, `Td_K`, `PAR_proportion`, `NIR_proportion`, `PAR_albedo`, or `NIR_albedo` in a batch/`targets_df` query fetches no dependencies and leaves the target variable uncomputed.
  * Additionally, `Tmin_C` is missing from `VARIABLE_TRANSFORMATIONS` in `constants.py` (unlike `Ta_C`), and `PAR_proportion` / `NIR_proportion` / `PAR_albedo` / `NIR_albedo` have no methods defined on `GEOS5FPConnection` (they were removed in `v2.24.0` while remaining listed in `COMPUTED_VARIABLES`, `README.md`, and `QUERYABLE_VARIABLES.md`).

### 1.3. Point Queries for `COMPUTED_VARIABLES` Fail in Single-Point and Time-Series Modes
* **Location:** `GEOS5FP/query.py` (lines 389–599) and `GEOS5FP/GEOS5FP_connection.py` (lines 1025–1048, 1085–1088)
* **Impact:** Querying computed variables such as `RH`, `VPD_kPa`, `Ea_Pa`, `SVP_Pa`, `Td_K`, or `wind_speed_mps` at a `Point`/`MultiPoint` or over a `time_range` fails or returns inconsistent types.
* **Details:**
  * In `GEOS5FP/query.py`, the `is_point_time_series` branch (line 395) and single-time `is_point_geometry` branch (line 513) check `if var_name in GEOS5FP_VARIABLES:` and otherwise raise `ValueError: Dataset must be specified when using raw variable name '...'`—they do not resolve `COMPUTED_VARIABLES` dependencies via OPeNDAP.
  * Calling `conn.RH(time_UTC=..., geometry=Point(...))` calls `self.Q(...)`, `self.PS(...)`, and `self.Ta_K(...)`, which return `np.ndarray` for points, and then calls `rt.clip(w / ws, 0, 1)` on a NumPy array (which fails or returns a `Raster` instead of `np.ndarray` when `rt.clip` is invoked on non-Raster arrays).

### 1.4. `albedo_visible` and `albedo_NIR` Formulas Do Not Match Documentation
* **Location:** `GEOS5FP/GEOS5FP_connection.py` (lines 1106–1118) and `GEOS5FP/query.py` (lines 192–193, 329–335)
* **Impact:** `albedo_visible` and `albedo_NIR` return raw `ALBVISDR` and `ALBNIRDR` directly rather than broadband visible/NIR albedo (`ALBVISDR + ALBVISDF` and `ALBNIRDR + ALBNIRDF`), and `albedo_visible` / `albedo_NIR` are not documented in `README.md` or `QUERYABLE_VARIABLES.md` (which still document `PAR_proportion`, `NIR_proportion`, `PAR_albedo`, and `NIR_albedo`).

---

## 2. Concurrency, Caching & Network Reliability Issues

### 2.1. Stale `.lock` Files Left Behind by `_granule_file_lock`
* **Location:** `GEOS5FP/GEOS5FP_connection.py` (`_granule_file_lock`, lines 63–90)
* **Impact:** Every downloaded granule leaves a persistent `<filename>.nc4.lock` file in the download directory.
* **Details:** `_granule_file_lock` creates `f"{destination_filename}.lock"` alongside the `.nc4` file and releases the `fcntl`/`msvcrt`/`FileLock` lock on exit, but never unlinks the `.lock` file. Directory listings and cleanup scripts scanning `~/data/GEOS5FP` encounter stray `.lock` files.

### 2.2. Unprotected `requests.head()` Calls Outside `create_robust_session`
* **Location:** `GEOS5FP/GEOS5FP_connection.py` (`_check_remote`, `is_year_available`, `months_available_in_year`, `is_month_available`, `dates_available_in_month`, `is_day_available`, `list_GEOS5FP_granules`, lines 164–285)
* **Impact:** Auxiliary metadata/availability methods call bare `requests.head(...)` with no timeout, no retry strategy, and no SSL fallback, risking indefinite hangs or transient `SSLError`/`ConnectionError` crashes when NASA NCCS servers stall.

### 2.3. `latest_time_available` Infinite Loop When `retries` Exhausted Without Exception
* **Location:** `GEOS5FP/GEOS5FP_connection.py` (`latest_time_available`, lines 246–257)
* **Impact:** Returns `None` silently if all 3 attempts fail rather than raising an exception, violating its `-> datetime` return annotation.

### 2.4. `query_geos5fp_point_multi` Mutates Global Socket Timeout Without Restoring It and Lacks Retry Logic
* **Location:** `GEOS5FP/GEOS5FP_point.py` (`query_geos5fp_point_multi`, lines 231–331)
* **Impact:** Unlike `query_geos5fp_point` (which saves/restores `socket.getdefaulttimeout()` in a `try/finally` block and retries transient `504 Gateway Timeout` / network errors), `query_geos5fp_point_multi` sets `socket.setdefaulttimeout(120)` globally without restoring it and performs zero retries on transient OPeNDAP failures.

### 2.5. Unused SSL Fallback Modules
* **Location:** `GEOS5FP/create_legacy_ssl_context.py` and `GEOS5FP/make_head_request_with_ssl_fallback.py`
* **Impact:** Dead code in the package. Neither module is imported anywhere in `GEOS5FP` after `v2.22.0` switched `download_file` to `create_robust_session` and `pooch`.

---

## 3. Spatial & Temporal Edge Cases

### 3.1. Multi-Variable Queries Not Supported for Raster Geometries
* **Location:** `GEOS5FP/query.py` (line 378)
* **Impact:** Passing multiple variables (e.g., `["Ta_K", "SM"]`) with a `RasterGeometry` or `RasterGrid` raises `ValueError: Multiple variables can only be queried for point geometries`, even though each variable can be retrieved as a `Raster` and returned as a `dict[str, Raster]` (as was briefly attempted in `v2.21.0` before being overwritten in `v2.24.0`).

### 3.2. Batch `query()` Uses Nearest-Neighbor Time Matching Instead of Temporal Interpolation
* **Location:** `GEOS5FP/query.py` (lines 272–281)
* **Impact:** When `query()` is called with a `targets_df` or sequence of `(time_UTC, geometry)` pairs, `temporal_interpolation="interpolate"` (the default) is ignored—batch mode always picks the single closest timestep (`closest_idx = time_diffs.argmin()`), whereas raster queries and single-point queries behave differently. For 3-hourly products (`CO2SC`, `AOT`, `U2M`, `V2M`), this introduces up to 1.5 hours of temporal offset instead of linear interpolation.

### 3.3. `MultiPoint` Geometries in Batch Queries Silently Discard All Points After Index 0
* **Location:** `GEOS5FP/query.py` (lines 147–148)
* **Impact:** If a `MultiPoint` is passed inside a `GeoSeries` or list of geometries in a batch query, `pt_lon, pt_lat = unwrapped_geom.geoms[0].x, unwrapped_geom.geoms[0].y` silently ignores all subsequent points in the `MultiPoint`.

### 3.4. `GEOS5FPGranule._read_via_netcdf4` Longitude Convention and 1-Pixel Grid Assumption
* **Location:** `GEOS5FP/GEOS5FP_granule.py` (`_read_via_netcdf4`, lines 85–107)
* **Impact:**
  * Assumes `lons` are already in `[-180, 180]` order without wrapping `[0, 360)` longitudes if encountered.
  * Hardcodes fallback grid spacing to `dx = 0.625`, `dy = 0.5` (0.625° × 0.5°) when `len(lons) <= 1`, whereas native GEOS-5 FP 2D collections are 0.3125° × 0.25°.

---

## 4. Packaging, Testing & Repository Hygiene Issues

### 4.1. Heavy Downstream Dependencies Declared as Core Runtime Requirements
* **Location:** `pyproject.toml` (lines 16–39)
* **Impact:** `ECOv002-calval-tables>=1.11.0` and `sentinel-tiles` are listed in `project.dependencies`, pulling calibration/validation table packages and Sentinel tiling dependencies into minimal `GEOS5FP` installations even though they are only used in standalone scripts/examples.

### 4.2. Root-Level Debug and One-Off Scripts Cluttering Repository Top Level
* **Location:** Repository root (`debug_COT_timeseries.py`, `generate_BESSJPL_GEOS5FP_inputs.py`, `generate_BESSJPL_GEOS5FP_inputs_single_day.py`, `generate_BESSJPL_GEOS5FP_inputs_with_albedo_visible.py`, `generate_FLiESANN_GEOS5FP_inputs.py`, `reproduce_geos5fp_download_timeout.py`, `test_numpy_return.py`, `test_query_single_variable.py`)
* **Impact:**
  * Two test files (`test_numpy_return.py` and `test_query_single_variable.py`) sit in the root directory outside `tests/` (excluded by `testpaths = tests` in `pytest.ini`) and make live network calls to NASA OPeNDAP servers without `@pytest.mark.network` markers or mocks.
  * Several `generate_*` scripts import external/non-existent helpers (`from check_distribution import check_distribution`) or reference outdated columns (`load_times_locations()`).

### 4.3. Broken `repomix` Exclude/Include Pattern in `makefile`
* **Location:** `makefile` (line 45)
* **Impact:** `npx repomix --ignore "**/*.csv,!GEOS5FP/variables.csv,..."` excludes `GEOS5FP/variables.csv` from repomix snapshots because `repomix` processes `--ignore` globs without negating earlier wildcard matches unless `--include` is used.

### 4.4. Deprecated `datetime.utcnow()` Usage
* **Location:** `GEOS5FP/GEOS5FP_connection.py` (line 232)
* **Impact:** `datetime.utcnow()` is deprecated in Python 3.12+ (and `pyproject.toml` targets `requires-python = ">=3.11"` with CI testing up to Python 3.14). Should use `datetime.now(timezone.utc).date()`.