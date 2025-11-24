# Point Query Refactoring Summary

## Overview

The GEOS5FP variable retrieval methods have been refactored to support **point queries** that return time-series data as pandas DataFrames, in addition to the existing raster functionality.

## Key Changes

### 1. **Dual Return Types**
- **Raster queries**: When `geometry` is a `RasterGeometry`, methods return a `Raster` object (existing behavior)
- **Point queries**: When `geometry` is a `Point` or `MultiPoint`, methods return a `pandas.DataFrame` with time-series data

### 2. **Supported Geometry Types**
Point queries work with:
- `shapely.geometry.Point` - single location
- `shapely.geometry.MultiPoint` - multiple locations
- Any geometry wrapper that contains these types

### 3. **OPeNDAP Integration**
- Point queries use the `query_geos5fp_point` function to access GEOS-5 FP data via OPeNDAP
- Faster than downloading full raster files for point locations
- Requires `xarray` and `netcdf4` packages (optional dependencies)

### 4. **Backward Compatibility**
- All existing raster functionality remains unchanged
- If xarray/netcdf4 are not installed, point queries will raise a helpful ImportError
- All tests pass with the refactored code

## Installation

### For Raster Queries Only (existing functionality)
```bash
# No additional requirements
```

### For Point Queries (new functionality)
```bash
conda install -c conda-forge xarray netcdf4
```

## Usage Examples

### Single Point Query

```python
from shapely.geometry import Point
from datetime import datetime
from GEOS5FP import GEOS5FPConnection

conn = GEOS5FPConnection()

# Create a point (longitude, latitude)
point = Point(-118.25, 34.05)  # Los Angeles
time_utc = datetime(2024, 6, 15, 12, 0)

# Query surface temperature - returns DataFrame
df = conn.Ts_K(time_UTC=time_utc, geometry=point)
print(df)
#                          Ts_K    lat     lon  lat_used  lon_used
# 2024-06-15 12:00:00  298.123  34.05 -118.25     34.0    -118.25
```

### Multiple Points Query

```python
from shapely.geometry import MultiPoint

# Query multiple locations at once
points = MultiPoint([
    (-118.25, 34.05),   # Los Angeles
    (-122.42, 37.77),   # San Francisco
    (-73.94, 40.73)     # New York
])

df = conn.Ta_K(time_UTC=time_utc, geometry=points)
print(df)
#                          Ta_K    lat     lon  lat_used  lon_used
# 2024-06-15 12:00:00  295.456  34.05 -118.25     34.0    -118.25
# 2024-06-15 12:00:00  290.123  37.77 -122.42     37.75   -122.5
# 2024-06-15 12:00:00  298.789  40.73  -73.94     40.75    -74.0
```

### Time-Series at a Point

```python
# For time-series, you can query the OPeNDAP server directly
# or call the method multiple times with different times
import pandas as pd

times = pd.date_range('2024-06-15', periods=24, freq='1H')
results = []

for t in times:
    df = conn.Ts_K(time_UTC=t, geometry=point)
    results.append(df)

time_series = pd.concat(results)
print(time_series)
```

### Regular Raster Query (unchanged)

```python
from rasters import RasterGeometry

# Define raster geometry
geometry = RasterGeometry.from_bounds(
    xmin=-120, ymin=30, xmax=-115, ymax=35,
    cell_size=0.25, crs="EPSG:4326"
)

# Returns a Raster object
raster = conn.Ts_K(time_UTC=time_utc, geometry=geometry)
print(type(raster))  # <class 'rasters.Raster'>
```

## Modified Methods

All simple variable retrieval methods now support point queries:

- `SFMC` / `SM` - Soil moisture
- `LAI` - Leaf area index
- `LHLAND` - Latent heat flux
- `EFLUX` - Latent heat flux
- `PARDR` / `PARDF` - PAR flux
- `AOT` - Aerosol optical thickness
- `COT` - Cloud optical thickness
- `Ts_K` - Surface temperature
- `Tmin_K` - Minimum air temperature
- `PS` - Surface pressure
- `Q` - Specific humidity
- `vapor_kgsqm` / `vapor_gccm` - Water vapor
- `ozone_dobson` / `ozone_cm` - Ozone
- `U2M` / `V2M` - Wind components
- `CO2SC` - CO2 concentration
- `SWin` / `SWTDN` - Shortwave radiation
- `ALBVISDR` / `ALBVISDF` / `ALBNIRDR` / `ALBNIRDF` / `ALBEDO` - Albedo

## DataFrame Output Format

Point query DataFrames include:
- **Index**: Timestamp(s) in UTC
- **Variable column**: The requested variable (e.g., 'Ts_K', 'Ta_K', 'SFMC')
- **lat**: Requested latitude
- **lon**: Requested longitude
- **lat_used**: Actual grid latitude used
- **lon_used**: Actual grid longitude used

## Implementation Details

### New Helper Methods

1. **`_is_point_geometry(geometry)`**
   - Detects if geometry is a Point or MultiPoint
   - Works with shapely and rasters geometry types

2. **`_extract_points(geometry)`**
   - Extracts (lat, lon) tuples from point geometries
   - Handles both Point and MultiPoint

3. **`_get_simple_variable(...)`** (modified)
   - Now checks if geometry is a point
   - Routes to OPeNDAP query for points
   - Routes to existing interpolation for rasters

### Optional Dependencies

The module gracefully handles missing optional dependencies:
```python
try:
    from .GEOS5FP_point import query_geos5fp_point
    HAS_OPENDAP_SUPPORT = True
except ImportError:
    HAS_OPENDAP_SUPPORT = False
```

If point query is attempted without xarray/netcdf4:
```python
ImportError: Point query support requires xarray and netCDF4. 
Install with: conda install -c conda-forge xarray netcdf4
```

## Testing

All existing tests pass:
```bash
$ python -m pytest tests/ -v
===================================================== 27 passed, 1 warning in 1.31s ======================================================
```

## Benefits

1. **Efficiency**: Point queries via OPeNDAP are much faster than downloading full raster files
2. **Flexibility**: Same API works for both raster and point queries
3. **Convenience**: Returns pandas DataFrame for easy time-series analysis
4. **Backward Compatible**: No breaking changes to existing code
5. **Optional**: Point query functionality is optional - doesn't require additional dependencies for raster-only use

## Future Enhancements

Potential improvements:
1. Add native time-range support for time-series queries
2. Support for polygon geometries with spatial averaging
3. Caching of OPeNDAP results
4. Parallel queries for multiple points
5. Integration with other meteorological data sources
