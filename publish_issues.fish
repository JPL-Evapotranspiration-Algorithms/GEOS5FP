#!/usr/bin/env fish

# publish_issues.fish
# Publishes the v2.26.0 pending updates (UPD-1 through UPD-8) to GitHub via the `gh` CLI.
# Skips creating an issue if an open issue with the exact same title already exists.

set -g REPO "JPL-Evapotranspiration-Algorithms/GEOS5FP"

if not type -q gh
    echo "Error: GitHub CLI (gh) is not installed or not in PATH." >&2
    exit 1
end

if not gh auth status >/dev/null 2>&1
    echo "Error: gh is not authenticated. Run 'gh auth login' first." >&2
    exit 1
end

set -g existing_titles (gh issue list --repo "$REPO" --state open --limit 500 --json title --jq '.[].title')

function ensure_label --argument-names name color description
    gh label create "$name" --repo "$REPO" --color "$color" --description "$description" --force >/dev/null 2>&1
end

function create_issue --argument-names title labels body
    if contains -- "$title" $existing_titles
        echo "Skipping (already open): $title"
        return 0
    end

    echo "Creating issue: $title"
    gh issue create \
        --repo "$REPO" \
        --title "$title" \
        --label "$labels" \
        --body "$body"
end

echo "Ensuring labels exist in $REPO..."
ensure_label "bug" "d73a4a" "Something isn't working"
ensure_label "enhancement" "a2eeef" "New feature or request"
ensure_label "high-priority" "b60205" "High-priority issue affecting downstream pipelines"
ensure_label "critical" "8b0000" "Critical bug causing silent data corruption or crashes"

# ------------------------------------------------------------------------------
# UPD-1 (Critical): Unpack MultiPoint geometries in spatio-temporal batch query()
# ------------------------------------------------------------------------------
create_issue \
    "Unpack MultiPoint geometries in spatio-temporal batch query()" \
    "bug,critical" \
'### Summary
When a `MultiPoint` of length $N$ is paired with a `time_UTC` sequence of length $N$ in `GEOS5FPConnection.query()` (such as when `JET3.process_JET_table()` runs in online mode), `is_batch_query` broadcasts `[geometry] * len(times)` and extracts `unwrapped_geom.geoms[0]` for every row—silently querying the first point’s coordinates for all $N$ timestamps.

### Location
`GEOS5FP/query.py` (`is_batch_query` geometry normalization)

### Proposed Fix
Unpack `list(unwrapped_input_geom.geoms)` when `len(pts) == len(times)`, broadcast when `len(pts) == 1`, and raise a descriptive `ValueError` on length mismatch.'

# ------------------------------------------------------------------------------
# UPD-2 (High): Writable fallback for Pooch staging root in read-only containers
# ------------------------------------------------------------------------------
create_issue \
    "Fall back to tempfile.gettempdir() when pooch.os_cache() is read-only" \
    "bug,high-priority" \
'### Summary
In Apptainer/Singularity containers on HPC nodes where `$HOME` (e.g., `/root/.cache/GEOS5FP`) is mounted read-only, `GEOS5FPConnection.download_file()` fails with `PermissionError` (mapped to `FailedGEOS5FPDownload`) when creating the Pooch staging directory.

### Location
`GEOS5FP/GEOS5FP_connection.py` (`download_file`)

### Proposed Fix
Add `_resolve_staging_root(download_root)` to probe `pooch.os_cache("GEOS5FP")` first and fall back to `Path(tempfile.gettempdir()) / "GEOS5FP_cache"` when `pooch.os_cache` cannot be created or is not writable.'

# ------------------------------------------------------------------------------
# UPD-3 (High): Explicit return_dict option on gridded query()
# ------------------------------------------------------------------------------
create_issue \
    "Add return_dict option to query() for gridded geometries" \
    "enhancement,high-priority" \
'### Summary
When `geometry` is a `RasterGrid` or `RasterGeometry`, `query(variables=["CO2SC", "Ta_K"], ...)` returns `dict[str, Raster]`, whereas `query(variables=["CO2SC"], ...)` unwraps the 1-element sequence and returns a `rasters.Raster` (whose `.keys` attribute is `None`). Callers expecting a dictionary mapping fail with `TypeError: '\''NoneType'\'' object is not callable`.

### Location
`GEOS5FP/query.py` (`query`)

### Proposed Fix
Add `return_dict: bool = False` to `query()`. Preserve `return_dict=False` by default so 1-element sequences return a `Raster`, and return `{var_name: raster}` when `return_dict=True` regardless of variable count.'

# ------------------------------------------------------------------------------
# UPD-4 (Medium): Honor allow_fallback=True on single-timestamp point queries
# ------------------------------------------------------------------------------
create_issue \
    "Honor allow_fallback=True on single-timestamp point queries in query()" \
    "bug" \
'### Summary
`GEOS5FPConnection.variable()` catches OPeNDAP failures and falls back to `self.interpolate(...)` when `self.allow_fallback` is `True`, whereas the single-timestamp `Point`/`MultiPoint` branch in `GEOS5FP/query.py` logs a warning and returns empty results without checking `conn.allow_fallback`.

### Location
`GEOS5FP/query.py` (single-time point query branch)

### Proposed Fix
When `query_geos5fp_point` fails or returns empty data and `conn.allow_fallback` is `True`, fall back to `conn.interpolate(...)` and apply `VARIABLE_TRANSFORMATIONS`.'

# ------------------------------------------------------------------------------
# UPD-5 (Medium): Add Td_C() method and register Td_C in COMPUTED_VARIABLES
# ------------------------------------------------------------------------------
create_issue \
    "Add Td_C() method and register Td_C in COMPUTED_VARIABLES" \
    "enhancement" \
'### Summary
`GEOS5FPConnection` provides `Ta_K()` + `Ta_C()` and `Tmin_K()` + `Tmin_C()`, but only `Td_K()` without `Td_C()`, forcing downstream models (such as `JET3.sharpen_meteorology_data`) to manually subtract `273.15`.

### Locations
- `GEOS5FP/GEOS5FP_connection.py`
- `GEOS5FP/constants.py`
- `GEOS5FP/query.py`

### Proposed Fix
Implement `GEOS5FPConnection.Td_C()`, register `"Td_C"` in `COMPUTED_VARIABLES` and `COMPUTED_DEPENDENCIES`, and evaluate `"Td_C"` in `_evaluate_computed_variable`.'

# ------------------------------------------------------------------------------
# UPD-6 (Medium): Route COMPUTED_VARIABLES inside GEOS5FPConnection.variable()
# ------------------------------------------------------------------------------
create_issue \
    "Route COMPUTED_VARIABLES in GEOS5FPConnection.variable()" \
    "bug" \
'### Summary
Calling `conn.variable("wind_speed_mps", ...)` or `conn.variable("Ta_C", ...)` raises `KeyError` in `get_variable_info()` because `GEOS5FPConnection.variable()` only checks `GEOS5FP_VARIABLES`.

### Location
`GEOS5FP/GEOS5FP_connection.py` (`GEOS5FPConnection.variable`)

### Proposed Fix
Check `if variable_name in COMPUTED_VARIABLES and hasattr(self, variable_name):` at the start of `GEOS5FPConnection.variable()` and delegate to the corresponding method.'

# ------------------------------------------------------------------------------
# UPD-7 (Low): Add deprecated wind_speed() compatibility wrapper
# ------------------------------------------------------------------------------
create_issue \
    "Add deprecated wind_speed() wrapper delegating to wind_speed_mps()" \
    "enhancement" \
'### Summary
`GEOS5FP >=2.24.0` standardized on `wind_speed_mps(...)` and removed `wind_speed(...)`, causing older external scripts or notebooks to fail with `AttributeError`.

### Location
`GEOS5FP/GEOS5FP_connection.py`

### Proposed Fix
Add a `wind_speed(...)` wrapper on `GEOS5FPConnection` that emits a `DeprecationWarning` and delegates to `self.wind_speed_mps(...)`.'

# ------------------------------------------------------------------------------
# UPD-8 (Low): Preserve point identity in MultiPoint time_range queries
# ------------------------------------------------------------------------------
create_issue \
    "Preserve point identity in MultiPoint time_range queries" \
    "bug" \
'### Summary
When `time_range=(start, end)` is queried with a `MultiPoint` of $M > 1$ locations and `single_variable=True`, `all_variable_dfs` concatenates each point’s series and sorts by timestamp—returning an unlabeled 1D `np.ndarray` of length $M \times T$ with rows interleaved across points.

### Location
`GEOS5FP/query.py` (`is_point_time_series` branch)

### Proposed Fix
Attach a `point_index` column and return the time-indexed `GeoDataFrame` containing `geometry` whenever `len(points) > 1` so multi-point time-series results remain unambiguous.'

echo "Done."
