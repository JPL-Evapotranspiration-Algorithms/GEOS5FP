# Multi-Variable Query Quick Reference

## Basic Syntax

```python
from GEOS5FP import GEOS5FPConnection

conn = GEOS5FPConnection()

# Single variable (str)
df = conn.variable("T2M", ...)

# Multiple variables (list)
df = conn.variable(["T2M", "PS", "QV2M"], ...)
```

## Common Use Cases

### 1. Single Point, Multiple Variables, Time Range
```python
df = conn.variable(
    ["T2M", "PS", "QV2M"],
    time_range=(start_time, end_time),
    dataset="tavg1_2d_slv_Nx",
    lat=35.799,
    lon=-76.656
)
```

### 2. Single Point, Multiple Variables, Single Time
```python
df = conn.variable(
    ["T2M", "PS", "QV2M"],
    time_UTC=datetime(2019, 10, 2, 19, 9, 40),
    dataset="tavg1_2d_slv_Nx",
    lat=35.799,
    lon=-76.656
)
```

### 3. Using Point Geometry
```python
from shapely.geometry import Point

point = Point(-76.656, 35.799)
df = conn.variable(
    ["T2M", "PS", "QV2M"],
    time_UTC=datetime(2019, 10, 2, 19, 9, 40),
    dataset="tavg1_2d_slv_Nx",
    geometry=point
)
```

### 4. Predefined Variables (No Dataset Needed)
```python
df = conn.variable(
    ["Ta_K", "SM", "LAI"],
    time_range=(start_time, end_time),
    lat=35.799,
    lon=-76.656
)
```

### 5. From GeoDataFrame
```python
import geopandas as gpd

gdf = gpd.read_file('locations.csv')
row = gdf.iloc[0]

df = conn.variable(
    ["T2M", "PS", "QV2M"],
    time_UTC=row['time_UTC'],
    dataset="tavg1_2d_slv_Nx",
    lat=row.geometry.y,
    lon=row.geometry.x
)
```

## Variable Groups by Dataset

### tavg1_2d_slv_Nx (Surface Variables)
```python
["T2M", "PS", "QV2M", "U2M", "V2M"]  # Meteorology
["SFMC", "GWETROOT", "GWETTOP"]       # Soil moisture
["LAI", "GRN"]                        # Vegetation
```

### tavg1_2d_rad_Nx (Radiation Variables)
```python
["SWGDN", "LWGAB", "LWGEM"]          # Solar & longwave
["PARDR", "PARDF"]                    # PAR components
```

### inst3_2d_asm_Nx (3-Hourly Variables)
```python
["T2M", "PS", "QV2M"]                 # Instantaneous values
```

## Output Format

```python
# Result DataFrame structure:
                     T2M         PS      QV2M    lat     lon  lat_used  lon_used
2019-10-02 19:00:00  295.5  101325.0  0.0125  35.799  -76.656    35.8    -76.7
2019-10-02 20:00:00  294.8  101320.0  0.0123  35.799  -76.656    35.8    -76.7
```

## Key Restrictions

✅ **Supported**: Multiple variables with point geometry  
❌ **Not Supported**: Multiple variables with raster geometry  

✅ **Supported**: Single variable with raster geometry  
✅ **Supported**: Single variable with point geometry  

## Error Handling

```python
try:
    df = conn.variable(
        ["T2M", "PS", "QV2M"],
        time_UTC=time,
        dataset="tavg1_2d_slv_Nx",
        lat=35.799,
        lon=-76.656
    )
except ValueError as e:
    print(f"Validation error: {e}")
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: conda install -c conda-forge xarray netcdf4")
```

## Best Practices

1. **Group by Dataset**: Query variables from the same dataset together
2. **Specify Dataset**: Always specify `dataset` when using raw variable names
3. **Use Predefined Names**: Use predefined variables when available (automatic dataset lookup)
4. **Handle Errors**: Wrap queries in try-except for production code
5. **Check Results**: Verify DataFrame has expected columns and data

## Performance Tips

- Querying multiple variables in one call is more efficient than separate calls
- Time range queries are faster than looping over individual times
- Point queries via OPeNDAP are much faster than downloading full granules

## See Full Documentation

- [MULTI_VARIABLE_GUIDE.md](MULTI_VARIABLE_GUIDE.md) - Complete usage guide
- [VARIABLE_METHOD_GUIDE.md](VARIABLE_METHOD_GUIDE.md) - Variable method details
- [GEOS5FP/VARIABLES_README.md](GEOS5FP/VARIABLES_README.md) - Available variables
