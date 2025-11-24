# Multi-Variable Query Feature

## Overview

The `variable()` method of `GEOS5FPConnection` now supports querying multiple variables simultaneously when using point geometries. This feature allows you to retrieve multiple meteorological variables in a single call, with each variable becoming a column in the output DataFrame.

## Key Features

- **Single or Multiple Variables**: Pass either a string for a single variable or a list of strings for multiple variables
- **Point Queries Only**: Multi-variable queries are supported for point geometries (single points or multiple points)
- **Efficient Merging**: Variables from the same dataset are queried together and merged into a single DataFrame
- **Flexible Input**: Works with both predefined variable names (e.g., "Ta_K") and raw GEOS-5 FP variable names (e.g., "T2M")

## Usage

### Single Variable (Original Behavior)

```python
from GEOS5FP import GEOS5FPConnection
from datetime import datetime, timedelta

conn = GEOS5FPConnection()
end_time = datetime(2019, 10, 2, 19, 9, 40)
start_time = end_time - timedelta(hours=6)

# Query a single variable
df = conn.variable(
    "T2M",
    time_range=(start_time, end_time),
    dataset="tavg1_2d_slv_Nx",
    lat=35.799,
    lon=-76.656
)
```

### Multiple Variables (New Feature)

```python
# Query multiple variables at once
df = conn.variable(
    ["T2M", "PS", "QV2M"],  # List of variables
    time_range=(start_time, end_time),
    dataset="tavg1_2d_slv_Nx",
    lat=35.799,
    lon=-76.656
)

# Result DataFrame has columns: T2M, PS, QV2M, lat, lon, lat_used, lon_used
print(df)
```

### Multiple Predefined Variables

```python
# Use predefined variable names (no dataset needed)
df = conn.variable(
    ["Ta_K", "SM", "LAI"],
    time_range=(start_time, end_time),
    lat=35.799,
    lon=-76.656
)
```

### Single Time, Multiple Variables

```python
# Query multiple variables at a single point in time
df = conn.variable(
    ["T2M", "PS", "QV2M"],
    time_UTC=datetime(2019, 10, 2, 19, 9, 40),
    dataset="tavg1_2d_slv_Nx",
    lat=35.799,
    lon=-76.656
)
```

### With GeoDataFrame

```python
import geopandas as gpd

# Load point locations from CSV
gdf = gpd.read_file('notebooks/spatio_temporal.csv')

# Get first location
first_record = gdf.iloc[0]
lat = first_record.geometry.y
lon = first_record.geometry.x
time_utc = first_record['time_UTC']

# Query multiple variables
df = conn.variable(
    ["T2M", "PS", "QV2M"],
    time_UTC=time_utc,
    dataset="tavg1_2d_slv_Nx",
    lat=lat,
    lon=lon
)
```

## Method Signature

```python
def variable(
    self,
    variable_name: Union[str, List[str]],  # Now accepts list!
    time_UTC: Union[datetime, str] = None,
    time_range: Tuple[Union[datetime, str], Union[datetime, str]] = None,
    dataset: str = None,
    geometry: RasterGeometry = None,
    resampling: str = None,
    lat: float = None,
    lon: float = None,
    dropna: bool = True,
    **kwargs
) -> Union[Raster, pd.DataFrame]
```

## Parameters

- **variable_name**: Now accepts either:
  - `str`: Single variable name (e.g., `"T2M"`)
  - `List[str]`: Multiple variable names (e.g., `["T2M", "PS", "QV2M"]`)
- Other parameters remain unchanged

## Return Value

When querying multiple variables with point geometry:
- Returns a `pd.DataFrame` with:
  - Index: datetime (time_UTC)
  - Columns: One column per variable, plus `lat`, `lon`, `lat_used`, `lon_used`

## Restrictions

1. **Point Geometries Only**: Multiple variables can only be queried for point geometries. Attempting to query multiple variables for raster geometries will raise a `ValueError`.

2. **Dataset Compatibility**: When querying multiple variables:
   - If all variables are predefined, their datasets are looked up automatically
   - If using raw variable names, you must provide the `dataset` parameter
   - All variables should ideally come from the same dataset for efficiency

## Examples from Real Use Cases

### Example 1: Meteorological Station Data

```python
# Query temperature, pressure, and humidity for a weather station
df = conn.variable(
    ["T2M", "PS", "QV2M", "U2M", "V2M"],
    time_range=(start_time, end_time),
    dataset="tavg1_2d_slv_Nx",
    lat=40.7128,
    lon=-74.0060  # New York City
)
```

### Example 2: Multiple Points with Multiple Variables

```python
from shapely.geometry import MultiPoint

# Define multiple locations
points = MultiPoint([(-118.25, 34.05), (-122.42, 37.77)])

df = conn.variable(
    ["T2M", "PS", "QV2M"],
    time_range=(start_time, end_time),
    dataset="tavg1_2d_slv_Nx",
    geometry=points
)
```

### Example 3: Batch Processing CSV Locations

```python
import geopandas as gpd
import pandas as pd

# Load locations
gdf = gpd.read_file('locations.csv')

results = []
for idx, row in gdf.iterrows():
    df = conn.variable(
        ["T2M", "PS", "QV2M", "SWGDN"],
        time_UTC=row['time_UTC'],
        dataset="tavg1_2d_slv_Nx",
        lat=row.geometry.y,
        lon=row.geometry.x
    )
    df['ID'] = row['ID']
    results.append(df)

# Combine all results
final_df = pd.concat(results, ignore_index=False)
```

## Available Variable Combinations

### Common Meteorological Variables (tavg1_2d_slv_Nx)
- Temperature: `"T2M"`, `"TLML"`
- Pressure: `"PS"`, `"SLP"`
- Humidity: `"QV2M"`, `"RH2M"`
- Wind: `"U2M"`, `"V2M"`, `"U10M"`, `"V10M"`

### Radiation Variables (tavg1_2d_rad_Nx)
- Solar: `"SWGDN"`, `"SWGNT"`, `"SWTDN"`
- Longwave: `"LWGAB"`, `"LWGEM"`
- PAR: `"PARDR"`, `"PARDF"`

### Land Surface Variables (tavg1_2d_slv_Nx)
- Soil: `"SFMC"`, `"GWETROOT"`, `"GWETTOP"`
- Vegetation: `"LAI"`, `"GRN"`
- Fluxes: `"LHLAND"`, `"SHLAND"`, `"EFLUX"`

## Error Handling

```python
try:
    df = conn.variable(
        ["T2M", "INVALID_VAR"],
        time_UTC=datetime.now(),
        dataset="tavg1_2d_slv_Nx",
        lat=40.0,
        lon=-120.0
    )
except ValueError as e:
    print(f"Variable not found: {e}")
except ImportError as e:
    print(f"Missing dependencies: {e}")
```

## Performance Considerations

1. **Efficient OPeNDAP Queries**: Each variable is queried separately but efficiently using OPeNDAP
2. **Parallel Processing**: Variables are queried sequentially but could be parallelized in future versions
3. **Memory Usage**: Each variable adds a column to the DataFrame, so memory usage scales linearly with the number of variables

## Migration Guide

If you have existing code that queries multiple variables separately:

**Before:**
```python
df1 = conn.variable("T2M", time_range=..., lat=lat, lon=lon)
df2 = conn.variable("PS", time_range=..., lat=lat, lon=lon)
df3 = conn.variable("QV2M", time_range=..., lat=lat, lon=lon)
# Manually merge dataframes
```

**After:**
```python
df = conn.variable(
    ["T2M", "PS", "QV2M"],
    time_range=...,
    lat=lat,
    lon=lon
)
# All variables in one DataFrame!
```

## See Also

- [Point Query Refactoring](POINT_QUERY_REFACTORING.md)
- [Variable Method Guide](VARIABLE_METHOD_GUIDE.md)
- [Variables README](GEOS5FP/VARIABLES_README.md)
