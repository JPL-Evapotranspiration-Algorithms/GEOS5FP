# Generalized Variable Retrieval Method

The `variable()` method provides a unified, user-friendly interface for retrieving any variable from any GEOS-5 FP dataset.

## Features

- **Flexible**: Works with both predefined variable names and raw GEOS-5 FP variable names
- **Efficient**: Uses OPeNDAP for fast time-series queries
- **Versatile**: Supports both raster and point queries
- **Simple**: Single method for all variable retrieval needs

## Basic Usage

### Time-Series Query (Point Location)

```python
from GEOS5FP import GEOS5FPConnection
from datetime import datetime, timedelta

conn = GEOS5FPConnection()

# Define time range
end_time = datetime(2024, 11, 15)
start_time = end_time - timedelta(days=7)

# Query air temperature time-series
df = conn.variable(
    "Ta_K",                          # Variable name
    time_range=(start_time, end_time),  # Time range
    lat=34.05,                       # Latitude
    lon=-118.25                      # Longitude
)
```

### Using Raw Variable Names

```python
# Use raw GEOS-5 FP variable name with explicit dataset
df = conn.variable(
    "T2M",                           # Raw variable name
    time_range=(start_time, end_time),
    dataset="tavg1_2d_slv_Nx",      # Must specify dataset
    lat=34.05,
    lon=-118.25
)
```

### Raster Query (Single Time)

```python
from rasters import RasterGeometry

# Define target area
geometry = RasterGeometry.open("target_area.tif")

# Query spatial data
raster = conn.variable(
    "Ta_K",
    time_UTC="2024-11-15 12:00:00",
    geometry=geometry,
    resampling="bilinear"
)
```

## Available Datasets

Common GEOS-5 FP datasets:
- `tavg1_2d_slv_Nx` - 1-hourly time-averaged, 2D, single-level diagnostics
- `tavg1_2d_rad_Nx` - 1-hourly time-averaged, 2D, radiation diagnostics
- `inst3_2d_asm_Nx` - 3-hourly instantaneous, 2D, single-level assimilation

## Common Variables

### Predefined Variables (no dataset required)
- `Ta_K` - Air temperature (Kelvin)
- `SM` / `SFMC` - Soil moisture
- `SWin` - Incoming shortwave radiation
- `U2M` - Eastward wind component
- `V2M` - Northward wind component
- `PS` - Surface pressure
- `Q` - Specific humidity

### Raw Variables (dataset required)
- `T2M` - 2-meter air temperature (tavg1_2d_slv_Nx)
- `QV2M` - 2-meter specific humidity (tavg1_2d_slv_Nx)
- `SWGDN` - Surface incoming shortwave flux (tavg1_2d_rad_Nx)
- `PS` - Surface pressure (tavg1_2d_slv_Nx)

## Multiple Variables Example

```python
# Query multiple variables and combine
df_temp = conn.variable("Ta_K", time_range=(start, end), lat=lat, lon=lon)
df_wind_u = conn.variable("U2M", time_range=(start, end), dataset="tavg1_2d_slv_Nx", lat=lat, lon=lon)
df_wind_v = conn.variable("V2M", time_range=(start, end), dataset="tavg1_2d_slv_Nx", lat=lat, lon=lon)

# Combine into single DataFrame
import pandas as pd
df = pd.DataFrame({
    'temperature': df_temp['Ta_K'] - 273.15,
    'wind_u': df_wind_u['U2M'],
    'wind_v': df_wind_v['V2M']
})
df['wind_speed'] = (df['wind_u']**2 + df['wind_v']**2)**0.5
```

## Parameters

- `variable_name` (str): Variable name (predefined or raw)
- `time_UTC` (datetime/str): Single time for raster queries
- `time_range` (tuple): (start, end) for time-series queries
- `dataset` (str): GEOS-5 FP dataset name (required for raw variables)
- `geometry` (RasterGeometry/Point/MultiPoint): Target geometry
- `lat`, `lon` (float): Point coordinates (alternative to geometry)
- `resampling` (str): Resampling method ("nearest", "bilinear", etc.)
- `dropna` (bool): Drop NaN values from results (default: True)

## Performance Tips

1. **Use time ranges for time-series**: Much faster than iterating
   ```python
   # Fast: single query
   df = conn.variable("Ta_K", time_range=(start, end), lat=lat, lon=lon)
   
   # Slow: multiple queries
   for time in time_steps:
       df = conn.variable("Ta_K", time_UTC=time, lat=lat, lon=lon)
   ```

2. **Combine multiple variables**: Query separately then merge
   ```python
   df1 = conn.variable("Ta_K", time_range=(start, end), lat=lat, lon=lon)
   df2 = conn.variable("U2M", time_range=(start, end), dataset="tavg1_2d_slv_Nx", lat=lat, lon=lon)
   ```

3. **Know your datasets**: Check which dataset contains your variable to avoid errors

## Requirements

- For point queries: `xarray` and `netcdf4` packages
  ```bash
  conda install -c conda-forge xarray netcdf4
  ```

## See Also

- `time_series_example.py` - Basic time-series example
- `variable_method_example.py` - Comprehensive examples
- `example_point_query.py` - Point query examples
