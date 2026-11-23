# Issue Report: `GEOS5FPConnection.query()` Rejects Single-Element List on `RasterGrid` Geometries

- **Date:** November 23, 2026
- **Package:** `GEOS5FP` (`>=2.24.0`)
- **Affected Downstream Consumers:** `BESS-JPL` (`~=1.28.0`), `JET3` (`~=3.1.1`), `ECOv003-L3T-L4T-JET` (`v1.22.1`)
- **Component:** `GEOS5FP/query.py` (`GEOS5FPConnection.query`, line 378)
- **Symptom:** `ValueError: Multiple variables can only be queried for point geometries` when passing a 1-element list (e.g., `["CO2SC"]`) with a `RasterGrid` / `RasterGeometry`.
- **Status:** Open / Proposed fix ready

---

## 1. Summary

When downstream callers invoke `GEOS5FPConnection.query()` with a sequence containing a single variable name (such as `["CO2SC"]`) and a gridded geometry (`rasters.RasterGrid`), `GEOS5FP` raises `ValueError: Multiple variables can only be queried for point geometries`.

Because only one variable is being queried, the call is semantically identical to passing a scalar string (`"CO2SC"`). Rejecting single-element sequences breaks downstream callers like `BESS-JPL` that construct a `variables_to_retrieve` list dynamically based on which inputs were not pre-supplied by the caller.

---

## 2. Reproduction & Traceback

Observed during end-to-end execution of `ECOv003-L3T-L4T-JET` (`examples/example_L3T_L4T_JET_run_with_download.py`) on tile `11SPS` (`2024-10-22T18:36:27Z`), where `JET3` supplies all GEOS-5 FP meteorological inputs except `CO2SC`, leaving `BESS_JPL` with `variables = ["CO2SC"]`:

```text
[2026-11-23 11:08:17 INFO] Retrieving GEOS-5 FP variables: CO2SC
[2026-11-23 11:08:17 INFO] Time UTC type: <class 'datetime.datetime'>
[2026-11-23 11:08:17 INFO] Geometry type: <class 'rasters.raster_grid.RasterGrid'>
Traceback (most recent call last):
  File "/Users/halverso/Projects/ECOv003-L3T-L4T-JET/examples/example_L3T_L4T_JET_run_with_download.py", line 61, in <module>
    exit_code = L3T_L4T_JET(runconfig_filename=runconfig_filename, overwrite=True, save_diagnostics=False)
  File "/Users/halverso/Projects/ECOv003-L3T-L4T-JET/ECOv003_L3T_L4T_JET/ECOv003_L3T_L4T_JET.py", line 133, in L3T_L4T_JET
    results = JET(
  File ".../JET3/JET.py", line 516, in JET
    BESS_results = BESS_JPL(
  File ".../BESS_JPL/model.py", line 131, in BESS_JPL
    BESS_inputs_dict = retrieve_BESS_inputs(
  File ".../BESS_JPL/retrieve_BESS_inputs.py", line 257, in retrieve_BESS_inputs
    GEOS5FP_inputs = retrieve_BESS_JPL_GEOS5FP_inputs(
  File ".../BESS_JPL/retrieve_BESS_JPL_GEOS5FP_inputs.py", line 189, in retrieve_BESS_JPL_GEOS5FP_inputs
    retrieved = GEOS5FP_connection.query(
  File ".../GEOS5FP/query.py", line 378, in query
    raise ValueError("Multiple variables can only be queried for point geometries")
ValueError: Multiple variables can only be queried for point geometries
```

Minimal standalone reproduction:

```python
from datetime import datetime
from GEOS5FP import GEOS5FP
from rasters import RasterGrid

conn = GEOS5FP()
grid = RasterGrid.from_bbox(xmin=-116.0, ymin=32.5, xmax=-115.0, ymax=33.5, cell_size=1000)
conn.query(variables=["CO2SC"], time_UTC=datetime(2024, 10, 22, 18, 30), geometry=grid)
# Raises ValueError: Multiple variables can only be queried for point geometries
```

---

## 3. Root Cause Analysis

1. `BESS_JPL/retrieve_BESS_JPL_GEOS5FP_inputs.py` (line 189) collects missing GEOS-5 FP variables into a Python `list` and passes it directly to `GEOS5FP_connection.query(variables=..., time_UTC=..., geometry=...)`.
2. In `GEOS5FP/query.py` (line 378), `query()` branches on container type (e.g., `isinstance(variables, (list, tuple))` or non-`str` iterable) rather than checking whether `len(variables) > 1`.
3. When `len(variables) == 1` and `geometry` is a `RasterGrid` (not a point/GeoDataFrame geometry), `query()` treats the 1-element list as a multi-variable request and raises `ValueError` instead of unwrapping `variables[0]` and returning a single-variable raster (or single-key mapping, matching its contract for scalar/1-variable queries).

---

## 4. Recommended Fix in `GEOS5FP`

### 4.1. Normalize Single-Element Sequences in `GEOS5FP/query.py`
In `GEOS5FPConnection.query()`:
- Check the length of `variables` when a non-string sequence/iterable is provided:
  - If `len(variables) == 0`: raise a descriptive `ValueError("At least one variable must be specified")`.
  - If `len(variables) == 1`: unwrap `var = variables[0]` and execute the standard single-variable query path (returning a `Raster` for raster geometries, or preserving the expected return structure if `return_dict` / `DataFrame` semantics apply).
  - Only raise `ValueError("Multiple variables can only be queried for point geometries")` when `len(variables) > 1` and `geometry` is not a point geometry.

### 4.2. Add Unit Test Coverage
Add a unit test in `GEOS5FP`'s test suite verifying that `query(variables=["CO2SC"], time_UTC=..., geometry=raster_grid)` succeeds and matches `query(variables="CO2SC", time_UTC=..., geometry=raster_grid)` without raising `ValueError`.