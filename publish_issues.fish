#!/usr/bin/env fish

# publish_issues.fish
# Publishes all known GEOS5FP issues to GitHub via the `gh` CLI.
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

# Fetch existing open issue titles to prevent duplicates
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
ensure_label "maintenance" "fbca04" "Code cleanup, dependency hygiene, and refactoring"

# ------------------------------------------------------------------------------
# 1. Single-element list & variables= alias rejected on RasterGrid geometries
# ------------------------------------------------------------------------------
create_issue \
    "GEOS5FPConnection.query() rejects single-element list and variables= alias on RasterGrid geometries" \
    "bug,high-priority" \
"### Summary
When callers invoke \`GEOS5FPConnection.query()\` with a single-element sequence (e.g., \`[\"CO2SC\"]\`) or pass \`variables=[\"CO2SC\"]\` alongside a gridded geometry (\`rasters.RasterGrid\` / \`RasterGeometry\`), \`GEOS5FP\` raises:

\`\`\`text
ValueError: Multiple variables can only be queried for point geometries
\`\`\`

This breaks downstream callers (\`BESS-JPL\`, \`JET3\`, \`ECOv003-L3T-L4T-JET\`) that dynamically build a list of missing variables.

### Root Cause
1. In \`GEOS5FP/query.py\`, \`single_variable\` is set via \`isinstance(target_variables, str)\` rather than \`len(variable_names) == 1\`.
2. \`variables=\` is not accepted as an alias for \`target_variables=\` / \`variable_name=\`.

### Proposed Fix
- Accept \`variables\` as an alias in \`GEOS5FP/query.py\`.
- Raise \`ValueError(\"At least one variable must be specified\")\` when an empty sequence is passed.
- Set \`single_variable = len(variable_names) == 1\` so 1-element sequences unwrap to the standard single-variable \`Raster\` query path.
- Add unit tests in \`tests/test_variable_queries.py\`."

# ------------------------------------------------------------------------------
# 2. Incomplete COMPUTED_VARIABLES evaluation in vectorized batch queries
# ------------------------------------------------------------------------------
create_issue \
    "Incomplete COMPUTED_VARIABLES evaluation in vectorized batch queries" \
    "bug,high-priority" \
"### Summary
\`COMPUTED_VARIABLES\` in \`GEOS5FP/constants.py\` registers 14 derived variables, but \`GEOS5FP/query.py\` only defines dependency mappings and formulas for 6 of them (\`RH\`, \`Ta_C\`, \`Tmin_C\`, \`wind_speed_mps\`, \`albedo_visible\`, \`albedo_NIR\`) during vectorized batch queries.

Querying any of the remaining 8 computed variables in batch / \`targets_df\` mode silently populates \`None\`:
- \`SVP_Pa\`
- \`Ea_Pa\`
- \`VPD_kPa\`
- \`Td_K\`
- \`PAR_proportion\`
- \`NIR_proportion\`
- \`PAR_albedo\`
- \`NIR_albedo\`

### Proposed Fix
Add complete dependency resolution (including transitive dependencies such as \`Ta -> SVP_Pa -> Ea_Pa -> VPD_kPa\`) and vectorized calculation formulas for all 14 entries in \`COMPUTED_VARIABLES\`."

# ------------------------------------------------------------------------------
# 3. Computed variables not supported in single-point or time-series OPeNDAP queries
# ------------------------------------------------------------------------------
create_issue \
    "Computed variables not supported in single-point or time-series OPeNDAP queries" \
    "bug,high-priority" \
"### Summary
Non-batch point queries (\`time_range=(start, end)\` or single-time \`Point\` / \`MultiPoint\`) in \`GEOS5FP/query.py\` look up every requested variable directly in \`GEOS5FP_VARIABLES\`.

Querying a computed variable such as \`Ta_C\`, \`RH\`, \`VPD_kPa\`, or \`wind_speed_mps\` for a single point or time series fails with:
\`\`\`text
ValueError: Dataset must be specified when using raw variable name '...'
\`\`\`

### Proposed Fix
Factor out a shared computed-variable dependency resolver and evaluator in \`GEOS5FP/query.py\` so single-point, multi-point, time-series, and batch queries all support \`COMPUTED_VARIABLES\` consistently."

# ------------------------------------------------------------------------------
# 4. Missing unit conversions in direct OPeNDAP point and batch queries
# ------------------------------------------------------------------------------
create_issue \
    "Missing unit conversions for vapor_gccm and ozone_cm in OPeNDAP point and batch queries" \
    "bug" \
"### Summary
Dedicated connection methods apply unit scaling after raster interpolation:
- \`vapor_gccm\` divides \`vapor_kgsqm\` (\`TQV\`) by \`10\` (\`kg/m^2\` -> \`g/cm^2\`)
- \`ozone_cm\` divides \`ozone_dobson\` (\`TO3\`) by \`1000\` (Dobson units -> \`cm\`)

However, \`VARIABLE_TRANSFORMATIONS\` in \`GEOS5FP/constants.py\` only defines \`Ta_C\`. As a result, \`conn.query(\"vapor_gccm\", ...)\` and \`conn.query(\"ozone_cm\", ...)\` via OPeNDAP return unscaled \`TQV\` and \`TO3\` values.

### Proposed Fix
Register \`vapor_gccm\` (\`lambda x: x / 10.0\`) and \`ozone_cm\` (\`lambda x: x / 1000.0\`) in \`VARIABLE_TRANSFORMATIONS\` and apply \`VARIABLE_TRANSFORMATIONS\` consistently across single-point, time-series, and batch query paths."

# ------------------------------------------------------------------------------
# 5. Nearest-neighbor temporal matching instead of interpolation in batch queries
# ------------------------------------------------------------------------------
create_issue \
    "Batch queries use nearest-neighbor matching instead of linear temporal interpolation" \
    "bug" \
"### Summary
\`GEOS5FPConnection.query()\` defaults to \`temporal_interpolation=\"interpolate\"\`, which matches the linear temporal interpolation behavior of raster queries. However, the vectorized batch branch in \`GEOS5FP/query.py\` selects the single closest timestamp (\`time_diffs.argmin()\`) within +/- 2 hours instead of interpolating between bracketing timesteps.

For 3-hourly products (\`tavg3_*\`, \`inst3_*\`), timestamps mid-way between granules can be up to 1.5 hours away from the selected value.

### Proposed Fix
Implement linear temporal interpolation between bracketing timestamps in the batch query branch when \`temporal_interpolation == \"interpolate\"\`, while preserving nearest-neighbor selection when \`temporal_interpolation == \"nearest\"\`."

# ------------------------------------------------------------------------------
# 6. Multi-variable raster queries not supported in query()
# ------------------------------------------------------------------------------
create_issue \
    "Support multi-variable raster queries in GEOS5FPConnection.query()" \
    "enhancement" \
"### Summary
Passing more than one variable (\`len(variables) > 1\`) with a \`RasterGrid\` / \`RasterGeometry\` raises:
\`\`\`text
ValueError: Multiple variables can only be queried for point geometries
\`\`\`

### Proposed Fix
When \`len(variable_names) > 1\` and \`geometry\` is a gridded geometry, retrieve each variable's \`Raster\` and return a \`dict[str, Raster]\` mapping variable names to their interpolated rasters."

# ------------------------------------------------------------------------------
# 7. Standalone GEOS5FP.query() call fails for raster queries
# ------------------------------------------------------------------------------
create_issue \
    "Standalone query() function call raises NotImplementedError for raster queries" \
    "bug" \
"### Summary
\`GEOS5FP/query.py\` attempts to support both bound method calls (\`conn.query(...)\`) and standalone calls (\`query(...)\`). When called standalone without an active \`GEOS5FPConnection\` instance, \`self\` is \`None\` and single-time raster queries fall through to \`NotImplementedError\`.

### Proposed Fix
Instantiate a default \`GEOS5FPConnection()\` when \`self is None\` in standalone \`query()\` calls (or clearly deprecate standalone invocation)."

# ------------------------------------------------------------------------------
# 8. Thread-unsafe global socket.setdefaulttimeout() mutation in GEOS5FP_point.py
# ------------------------------------------------------------------------------
create_issue \
    "Avoid mutating global socket.setdefaulttimeout() in GEOS5FP_point.py" \
    "bug" \
"### Summary
In \`GEOS5FP/GEOS5FP_point.py\`:
1. \`_query_geos5fp_point_impl()\` mutates process-wide \`socket.setdefaulttimeout(120.0)\`, which is not thread-safe under concurrent execution.
2. \`query_geos5fp_point_multi()\` sets \`socket.setdefaulttimeout(120)\` without a \`try...finally\` block to restore the original timeout, and lacks the retry/backoff logic present in \`query_geos5fp_point()\`.

### Proposed Fix
- Wrap \`query_geos5fp_point_multi()\` in the same retry/backoff loop used by \`query_geos5fp_point()\`.
- Scope timeout configuration per-request or restore the prior timeout safely in \`finally\` blocks."

# ------------------------------------------------------------------------------
# 9. Unbounded in-memory directory listing cache misses new near-real-time granules
# ------------------------------------------------------------------------------
create_issue \
    "Add TTL or cache invalidation to GEOS5FPConnection._listings" \
    "bug" \
"### Summary
\`GEOS5FPConnection.list_remote_directory()\` caches HTTP directory listings in \`self._listings[URL]\` for the lifetime of the connection object with no expiration or invalidation mechanism.

In long-running processes or services querying near-real-time data for the current UTC day, newly published granules on \`portal.nccs.nasa.gov\` will not be discovered once the directory has been listed once.

### Proposed Fix
Add a configurable TTL (e.g., 5–15 minutes for recent dates) or a \`refresh: bool = False\` parameter to \`list_remote_directory()\`."

# ------------------------------------------------------------------------------
# 10. Dead save_products parameter and unhardened requests.head() calls
# ------------------------------------------------------------------------------
create_issue \
    "Remove dead save_products parameter and use robust session for directory HEAD checks" \
    "maintenance" \
"### Summary
1. \`GEOS5FPConnection.__init__\` accepts \`save_products: bool = False\` and assigns \`self.save_products\`, but intermediate GeoTIFF caching was removed in \`v1.1.1\` and the attribute is no longer used anywhere in the codebase.
2. Availability helpers (\`is_year_available\`, \`is_month_available\`, \`is_day_available\`, \`months_available_in_year\`, \`dates_available_in_month\`, \`list_GEOS5FP_granules\`) call bare \`requests.head()\` without retries or SSL fallback handling.

### Proposed Fix
- Deprecate or remove \`save_products\`.
- Route \`HEAD\` checks through \`create_robust_session()\` / \`make_head_request_with_ssl_fallback()\`."

# ------------------------------------------------------------------------------
# 11. Relocate root-level debug scripts and live-network test_*.py files
# ------------------------------------------------------------------------------
create_issue \
    "Move root-level debug scripts to examples/ and convert root test_*.py into mocked unit tests" \
    "maintenance" \
"### Summary
Several ad-hoc scripts and live-network test files live at the repository root:
- \`test_numpy_return.py\`
- \`test_query_single_variable.py\`
- \`debug_COT_timeseries.py\`
- \`reproduce_geos5fp_download_timeout.py\`
- \`generate_BESSJPL_GEOS5FP_inputs.py\`
- \`generate_BESSJPL_GEOS5FP_inputs_single_day.py\`
- \`generate_BESSJPL_GEOS5FP_inputs_with_albedo_visible.py\`
- \`generate_FLiESANN_GEOS5FP_inputs.py\`

Running \`pytest test_numpy_return.py\` or executing these files directly triggers live OPeNDAP network requests.

### Proposed Fix
Move diagnostic/generation scripts into \`examples/\` (or \`scripts/\`) and migrate \`test_numpy_return.py\` and \`test_query_single_variable.py\` into \`tests/\` using mocked OPeNDAP responses."

# ------------------------------------------------------------------------------
# 12. Move script-only packages out of core runtime dependencies in pyproject.toml
# ------------------------------------------------------------------------------
create_issue \
    "Trim script-only packages from core runtime dependencies in pyproject.toml" \
    "maintenance" \
"### Summary
\`pyproject.toml\` currently includes several packages in core \`dependencies\` that are not imported anywhere inside the \`GEOS5FP/\` library package:
- \`ECOv002-calval-tables\` (only used in root \`generate_*.py\` scripts)
- \`sentinel-tiles\` (only used in \`examples/\`)
- \`pytictoc\` (unused; package uses \`GEOS5FP.timer.Timer\`)
- \`dask\` (unused)

### Proposed Fix
Move \`ECOv002-calval-tables\` and \`sentinel-tiles\` to an optional dependency group (e.g., \`[project.optional-dependencies] examples = [...]\`) and remove unused \`pytictoc\` and \`dask\` entries."

echo "Done."