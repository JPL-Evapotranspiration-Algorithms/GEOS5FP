#!/usr/bin/env fish

set -l repo "JPL-Evapotranspiration-Algorithms/GEOS5FP"
set -l title "Regression: vapor_gccm and ozone_cm return raw TQV/TO3 units on gridded raster queries"
set -l body_file (mktemp -t geos5fp_issue.XXXXXX.md)

function cleanup --on-event fish_exit
    if test -n "$body_file" -a -f "$body_file"
        rm -f "$body_file"
    end
end

begin
    echo '## Summary'
    echo ''
    echo 'In `GEOS5FP` (`v2.24.0`–`v2.26.0`), gridded raster queries for `vapor_gccm` and `ozone_cm` return raw GEOS-5 FP units instead of applying the unit conversions defined in `VARIABLE_TRANSFORMATIONS`:'
    echo ''
    echo '- **`vapor_gccm()`** returns raw `TQV` in $\text{kg m}^{-2}$ ($\text{mm}$) instead of dividing by `10.0` to return $\text{g cm}^{-2}$ (**10× too high**).'
    echo '- **`ozone_cm()`** returns raw `TO3` in Dobson Units ($\text{DU}$) instead of dividing by `1000.0` to return $\text{cm}$ / $\text{atm-cm}$ (**1000× too high**).'
    echo ''
    echo '## Root Cause'
    echo ''
    echo '`VARIABLE_TRANSFORMATIONS` in `GEOS5FP/constants.py` defines:'
    echo ''
    echo '```python'
    echo 'VARIABLE_TRANSFORMATIONS = {'
    echo '    "Ta_C": lambda x: x - 273.15,'
    echo '    "Tmin_C": lambda x: x - 273.15,'
    echo '    "Td_C": lambda x: x - 273.15,'
    echo '    "vapor_gccm": lambda x: x / 10.0,'
    echo '    "ozone_cm": lambda x: x / 1000.0,'
    echo '}'
    echo '```'
    echo ''
    echo 'However, `VARIABLE_TRANSFORMATIONS` is only applied in `GEOS5FP/query.py` (OPeNDAP point/batch paths) and `GEOS5FPConnection._query_point_via_opendap()`. When `GEOS5FPConnection.vapor_gccm()` or `GEOS5FPConnection.ozone_cm()` is called with a gridded `RasterGeometry` (or `geometry=None`), `.query()` routes to `GEOS5FPConnection.variable()` -> `GEOS5FPConnection.interpolate()`, which returns raw `TQV` and `TO3` without applying `VARIABLE_TRANSFORMATIONS`.'
    echo ''
    echo '## Downstream Impact'
    echo ''
    echo 'Confirmed during end-to-end validation of `ECOv003-L3T-L4T-JET v1.23.0` (`JET3 v3.3.0`, `FLiESANN v2.13.1`) on orbit `44841`, scene `001`, tile `11SPS` (`2026-06-02T18:00:12Z`):'
    echo ''
    echo '- Passing unscaled `vapor_gccm` (`mean = 10.7180` instead of `1.0718 g/cm²`) and `ozone_cm` (`mean = 334.4433` instead of `0.3344 cm`) into `FLiESANN` shifted predicted `SWin_Wm2` (`Rg` in `L3T_ETAUX`) by **`+74.35 W/m²`** (`RMSE = 74.757 W/m²`, $r = 0.38894$).'
    echo '- Applying `vapor_gccm / 10.0` and `ozone_cm / 1000.0` restores agreement with the public Collection 3 reference product (`RMSE = 0.527 W/m²`, `Bias = -0.023 W/m²`, $r = 0.99859$).'
    echo ''
    echo '## Proposed Fix'
    echo ''
    echo '1. Apply `VARIABLE_TRANSFORMATIONS[variable_name]` in `GEOS5FPConnection.variable()` for raster / `interpolate()` results (before `clip_min`/`clip_max`), ensuring `vapor_gccm()` and `ozone_cm()` return $\text{g cm}^{-2}$ and $\text{cm}$ consistently across raster, point, time-series, and batch queries without double-applying transformations on OPeNDAP paths.'
    echo '2. Add unit-range regression tests in `tests/test_variable_queries.py` verifying that gridded `Raster` queries for `vapor_gccm` (`TQV / 10.0`) and `ozone_cm` (`TO3 / 1000.0`) apply the expected unit conversions.'
end > "$body_file"

gh issue create \
    --repo "$repo" \
    --title "$title" \
    --label "bug" \
    --body-file "$body_file"

set -l exit_status $status
rm -f "$body_file"
exit $exit_status