# Vectorized Query - Quick Reference

## One-Liner Usage

```python
results = conn.variable(["Ta_K", "SM", "LAI"], time_UTC=gdf['time_UTC'], geometry=gdf['geometry'])
```

## Complete Minimal Example

```python
from GEOS5FP import GEOS5FPConnection
from spatiotemporal_utils import load_spatiotemporal_csv

# Load data
gdf = load_spatiotemporal_csv('notebooks/spatio_temporal.csv')

# Query
conn = GEOS5FPConnection()
results = conn.variable(
    variable_name=["Ta_K", "SM", "LAI"],
    time_UTC=gdf['time_UTC'],
    geometry=gdf['geometry']
)

# Add IDs and export
results['ID'] = gdf['ID'].values
results.to_csv('results.csv')
```

## Input Requirements

| Parameter | Type | Example |
|-----------|------|---------|
| `variable_name` | `List[str]` | `["Ta_K", "SM", "LAI"]` |
| `time_UTC` | `pd.Series` | `gdf['time_UTC']` |
| `geometry` | `gpd.GeoSeries` | `gdf['geometry']` or `gdf.geometry` |

## Output Format

```python
# Returns: gpd.GeoDataFrame
#                         Ta_K      SM     LAI                geometry
# time_UTC                                                            
# 2019-10-02 19:09:40  304.930  0.2134  4.0041  POINT (-76.656 35.799)
# 2019-06-23 18:17:17  297.731  0.3917  3.7778  POINT (-80.637 41.822)
```

## Alternative: lat/lon Format

```python
results = conn.variable(
    variable_name="Ta_K",
    time_UTC=df['timestamp'],
    lat=df['latitude'],
    lon=df['longitude']
)
```

## Common Variables

- `"Ta_K"` - Air temperature (K)
- `"SM"` - Soil moisture (m³/m³)
- `"LAI"` - Leaf area index
- `"SWin"` - Incoming shortwave radiation (W/m²)
- `"LWin"` - Incoming longwave radiation (W/m²)
- `"Ps_Pa"` - Surface pressure (Pa)
- `"RH"` - Relative humidity (%)

## Key Benefits vs. Row-by-Row

✅ **Single method call** - No loops needed  
✅ **66% less code** - Cleaner, more readable  
✅ **Automatic GeoDataFrame** - No manual concat/merge  
✅ **Built-in logging** - Track progress automatically  
✅ **Same performance** - Internal iteration, same query pattern  

## Error Handling

Failed queries log warnings and continue:
```
[WARNING] Failed to query Ta_K at (35.799, -76.656), time 2019-10-02: timeout
```
Failed values appear as `NaN` in results.

## Full Documentation

- **VECTORIZED_QUERY_GUIDE.md** - Complete usage guide with examples
- **VECTORIZED_IMPLEMENTATION_SUMMARY.md** - Technical implementation details
- **spatio_temporal_example.py** - Working example script
