# GEOS-5 FP Consolidated Variable Retrieval System

## Overview

The GEOS-5 FP connection class has been enhanced with a consolidated variable retrieval system that:

1. **Eliminates code duplication** across ~40+ variable methods
2. **Enables time-series generation** when `time_UTC` is a list
3. **Supports flexible geometry types** for spatial sampling
4. **Maintains full backward compatibility** with existing code

## Key Features

### 1. Unified Variable Access

All GEOS-5 FP variables are now accessible through a single consolidated method:

```python
result = geos5fp.get_variable(
    variable_name='SFMC',
    time_UTC='2023-01-01 12:00',
    geometry=None,
    resampling=None
)
```

### 2. Time-Series Support

Pass a list of times to automatically generate time-series data:

```python
# Time-series with multiple timestamps
times = ['2023-01-01 12:00', '2023-01-02 12:00', '2023-01-03 12:00']
ts_data = geos5fp.SFMC(times)  # Returns pandas DataFrame
```

### 3. Flexible Geometry Types

The `geometry` parameter now accepts multiple types:

- **RasterGeometry** (original): Returns raster data
- **List of Points**: Returns point-sampled values
- **MultiPoint**: Returns point-sampled values  
- **Coordinate tuples**: `[(lon, lat), ...]` format

```python
from shapely.geometry import Point, MultiPoint

# Point extraction examples
points = [Point(-118.25, 34.05), Point(-74.01, 40.71)]
coords = [(-118.25, 34.05), (-74.01, 40.71)]
multipoint = MultiPoint([(-118.25, 34.05), (-74.01, 40.71)])

# All return point-sampled values
values1 = geos5fp.SFMC('2023-01-01 12:00', geometry=points)
values2 = geos5fp.SFMC('2023-01-01 12:00', geometry=coords)  
values3 = geos5fp.SFMC('2023-01-01 12:00', geometry=multipoint)
```

### 4. Combined Time-Series and Point Extraction

Generate time-series of point-sampled data:

```python
times = ['2023-01-01 12:00', '2023-01-02 12:00', '2023-01-03 12:00']
points = [Point(-118.25, 34.05), Point(-74.01, 40.71)]

# Returns DataFrame: times as rows, points as columns
ts_points = geos5fp.SFMC(times, geometry=points)
```

## Return Type Matrix

| Time Input | Geometry Input | Return Type | Description |
|------------|----------------|-------------|-------------|
| Single | RasterGeometry | Raster | Single raster |
| Single | Points | pd.Series | Values at points |
| List | RasterGeometry | pd.DataFrame | Time-series of rasters |
| List | Points | pd.DataFrame | Time-series of point values |

## Variable Configuration Registry

All variables are centrally configured in `GEOS5FP_VARIABLES`:

```python
GEOS5FP_VARIABLES = {
    'SFMC': {
        'name': 'top layer soil moisture',
        'product': 'tavg1_2d_lnd_Nx',
        'variable': 'SFMC',
        'interval': 1,
        'min_value': 0,
        'max_value': 1,
        'exclude_values': [1],
        'cmap': 'SM_CMAP'
    },
    # ... more variables
}
```

## Available Variables

The system supports these GEOS-5 FP variables:

- **Land/Surface**: `SFMC`, `LAI`, `LHLAND`, `TS`
- **Atmospheric**: `T2M`, `T2MMIN`, `PS`, `QV2M`, `TQV`, `TO3`, `U2M`, `V2M`, `CO2SC`
- **Radiation**: `PARDR`, `PARDF`, `AOT`, `COT`, `SWGNT`, `SWTDN`
- **Albedo**: `ALBVISDR`, `ALBVISDF`, `ALBNIRDF`, `ALBNIRDR`, `ALBEDO`
- **Derived**: `EFLUX` (and others through existing methods)

## Multi-Variable Time-Series

Generate time-series for multiple variables simultaneously:

```python
# Get multiple variables for the same times/locations
multi_var = geos5fp.get_time_series(
    variable_names=['SFMC', 'LAI', 'TS'],
    time_UTC=['2023-01-01 12:00', '2023-01-02 12:00'],
    geometry=points
)
```

## Backward Compatibility

All existing method signatures work exactly as before:

```python
# These still work unchanged
sfmc = geos5fp.SFMC('2023-01-01 12:00')
lai = geos5fp.LAI('2023-01-01 12:00', geometry=target_geometry)
ts_k = geos5fp.Ts_K('2023-01-01 12:00', resampling='bilinear')
```

## Performance Benefits

- **Parallel processing** for multiple time points (configurable `max_workers`)
- **Granule caching** reduces duplicate downloads
- **Batch operations** optimize network usage
- **Efficient point sampling** using raster.sample()

## Error Handling

The system provides comprehensive error handling:

- **Unknown variables**: Clear error messages with available options
- **Invalid geometries**: Automatic type detection and conversion
- **Missing data**: Graceful handling with NaN values
- **Network issues**: Retry logic and detailed error reporting

## Migration Guide

### For Simple Cases
No changes needed - existing code continues to work.

### For Time-Series
Replace loops with list inputs:

```python
# Old approach
results = []
for time in time_list:
    result = geos5fp.SFMC(time)
    results.append(result)

# New approach  
results = geos5fp.SFMC(time_list)  # Returns DataFrame directly
```

### For Point Extraction
Use geometry parameter with point objects:

```python
# New point extraction
points = [Point(lon, lat) for lon, lat in coordinates]
values = geos5fp.SFMC('2023-01-01 12:00', geometry=points)
```

## Implementation Details

### Core Methods

1. **`get_variable()`**: Main consolidated retrieval method
2. **`get_time_series()`**: Multi-variable time-series helper  
3. **`_normalize_geometry()`**: Geometry type handling
4. **`_extract_at_points()`**: Point sampling logic

### Thread Safety
The parallel processing uses ThreadPoolExecutor for thread-safe operations.

### Memory Management
Time-series operations use efficient DataFrame structures and optional result caching.

## Examples

See `example_consolidated_usage.py` for comprehensive usage examples covering all supported patterns.