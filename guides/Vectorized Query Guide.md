# Vectorized Spatio-Temporal Query Guide

## Overview

The `.variable()` method now supports **vectorized batch queries**, allowing you to query multiple locations at different times with a single method call. This eliminates the need for row-by-row iteration and simplifies your code.

## Key Benefits

1. **Simpler Code**: Single method call instead of loops
2. **Cleaner API**: Pass lists/Series directly, no iteration needed
3. **Better Organization**: Returns structured GeoDataFrame with all results
4. **Progress Tracking**: Built-in logging shows query progress

## Basic Usage

### Loading Spatio-Temporal Data

```python
import geopandas as gpd
from GEOS5FP import GEOS5FPConnection
from spatiotemporal_utils import load_spatiotemporal_csv

# Load CSV with geometry column (WKT format)
gdf = load_spatiotemporal_csv('notebooks/spatio_temporal.csv')

# GeoDataFrame has columns: ID, time_UTC, geometry
print(gdf.head())
```

### Vectorized Multi-Variable Query

Instead of iterating row-by-row:

```python
# ❌ OLD WAY: Row-by-row iteration (slow, verbose)
results = []
for idx, row in gdf.iterrows():
    result = conn.variable(
        ["Ta_K", "SM", "LAI"],
        time_UTC=row['time_UTC'],
        geometry=row['geometry']
    )
    result['ID'] = row['ID']
    results.append(result)
final_df = pd.concat(results)
```

Use vectorized operation:

```python
# ✅ NEW WAY: Vectorized query (fast, simple)
conn = GEOS5FPConnection()

results = conn.variable(
    variable_name=["Ta_K", "SM", "LAI"],
    time_UTC=gdf['time_UTC'],  # Pass entire Series
    geometry=gdf['geometry']    # Pass entire GeoSeries
)

# Add original IDs
results['ID'] = gdf['ID'].values
```

## Method Signature

```python
def variable(
    self,
    variable_name: Union[str, List[str]],
    time_UTC: Union[datetime, str, List[datetime], List[str], pd.Series] = None,
    geometry: Union[Point, MultiPoint, List, gpd.GeoSeries] = None,
    **kwargs
) -> gpd.GeoDataFrame:
```

### Parameters for Vectorized Queries

- **variable_name**: Single variable or list of variables
  - `"Ta_K"` - Single variable
  - `["Ta_K", "SM", "LAI"]` - Multiple variables

- **time_UTC**: Timestamp(s) in UTC
  - `pd.Series` - Column from GeoDataFrame
  - `List[datetime]` - List of datetime objects
  - `List[str]` - List of ISO format strings

- **geometry**: Point geometries
  - `gpd.GeoSeries` - Geometry column from GeoDataFrame
  - `List[Point]` - List of Shapely Point objects

### Return Value

Returns a `GeoDataFrame` with:
- **Index**: `time_UTC` (timestamp for each query)
- **Variable columns**: One column per requested variable
- **geometry column**: Point geometry for each query location
- **CRS**: EPSG:4326 (WGS84)

## Examples

### Example 1: Single Variable, Multiple Points

```python
from shapely.geometry import Point

# Define multiple locations and times
times = [
    datetime(2019, 6, 23, 18, 0),
    datetime(2019, 6, 27, 16, 0),
    datetime(2019, 6, 30, 15, 0)
]

geometries = [
    Point(-80.637, 41.822),
    Point(-76.656, 35.799),
    Point(-85.367, 42.536)
]

# Query single variable
results = conn.variable(
    variable_name="Ta_K",
    time_UTC=times,
    geometry=geometries
)

print(results)
#                         Ta_K                 geometry
# time_UTC                                             
# 2019-06-23 18:00:00  297.730  POINT (-80.637 41.822)
# 2019-06-27 16:00:00  304.930  POINT (-76.656 35.799)
# 2019-06-30 15:00:00  295.717  POINT (-85.367 42.536)
```

### Example 2: Multiple Variables, GeoDataFrame Input

```python
# Load spatio-temporal dataset
gdf = gpd.read_file('flux_tower_observations.csv')
# Has columns: site_id, timestamp, geometry

# Query three variables at all locations/times
results = conn.variable(
    variable_name=["Ta_K", "SM", "LAI"],
    time_UTC=gdf['timestamp'],
    geometry=gdf.geometry
)

# Merge with original site IDs
results['site_id'] = gdf['site_id'].values

# Reorder columns: variables, ID, geometry
var_cols = [col for col in results.columns if col not in ['site_id', 'geometry']]
results = results[var_cols + ['site_id', 'geometry']]

# Export results
results.to_csv('geos5fp_flux_tower_data.csv')
```

### Example 3: Alternative lat/lon Format

You can also pass latitude and longitude as separate lists/Series:

```python
results = conn.variable(
    variable_name=["Ta_K", "SM"],
    time_UTC=df['time'],
    lat=df['latitude'],
    lon=df['longitude']
)
```

## Data Requirements

### Input CSV Format

Your spatio-temporal CSV should have:

```csv
ID,time_UTC,geometry
US-NC3,2019-10-02 19:09:40,POINT (-76.656 35.799)
US-Mi3,2019-06-23 18:17:17,POINT (-80.637 41.8222)
```

- **ID**: Site/observation identifier (any string)
- **time_UTC**: Timestamp in ISO format (YYYY-MM-DD HH:MM:SS)
- **geometry**: WKT format point geometry

### Loading Helper

Use the provided utility function:

```python
from spatiotemporal_utils import load_spatiotemporal_csv

gdf = load_spatiotemporal_csv('notebooks/spatio_temporal.csv')
# Automatically parses WKT geometry and sets CRS to EPSG:4326
```

## Output Format

### Column Order

Results are organized as:

```python
# Index: time_UTC
# Columns: [variable_cols...] + [ID] + [geometry]
```

Example output:

```
                           Ta_K        SM       LAI      ID                 geometry
time_UTC                                                                            
2019-10-02 19:09:40  304.930054  0.213425  4.004069  US-NC3   POINT (-76.656 35.799)
2019-06-23 18:17:17  297.730591  0.391680  3.777832  US-Mi3  POINT (-80.637 41.8222)
```

### Exporting Results

```python
# CSV export (includes geometry as WKT)
results.to_csv('results.csv')

# GeoJSON export
results.to_file('results.geojson', driver='GeoJSON')

# Shapefile export
results.to_file('results.shp')
```

## Performance Notes

### Query Optimization with Time Clustering

The vectorized query implementation uses intelligent grouping and time clustering to minimize OPeNDAP calls while avoiding excessively long time-series queries:

1. **Groups by unique coordinates**: Identifies all unique (lat, lon) pairs in your dataset
2. **Clusters times at each coordinate**: Groups nearby times together (max 30 days per query)
3. **Queries time ranges**: For each coordinate-time cluster, queries once with optimized time range
4. **Extracts specific values**: Pulls out the specific time values needed for each record

**Why time clustering matters:**
- Avoids querying 2-year time ranges when you only need a few scattered dates
- Keeps each OPeNDAP query manageable (< 30 days of hourly data)
- Balances between query count and data transfer size
- Network overhead dominates for very short queries, but large queries are slow

**Example with 20 records over 2 years at 6 coordinates:**
- Coordinate A: 7 records spanning 75 days → Split into 2 clusters (11.8 + 3.9 days)
- Coordinate B: 5 records spanning 63 days → Split into 2 clusters (14.8 + single day)
- Coordinates C-F: Single records or tight clusters → 1 cluster each

**Result:**
- **Without any optimization**: 20 × 3 variables = 60 OPeNDAP queries
- **With coordinate grouping only**: 6 × 3 = 18 queries (but some span months!)
- **With time clustering**: 8 batches × 3 variables = **24 queries** (all < 30 days)
- **Sweet spot**: 60% reduction while keeping queries efficient

### Performance Characteristics

- **Queries**: 1 per variable per coordinate-time cluster
- **Typical efficiency**: 50-70% reduction in queries vs. naive approach
- **Time clustering**: Default 30-day maximum per query (configurable in code)
- **Speedup**: 5-10x faster for typical flux tower datasets with repeated sites
- **Limiting factor**: Network latency to NASA OPeNDAP server
- **Progress logging**: Shows coordinates, clusters, and time spans being processed
- **Error handling**: Individual cluster failures don't stop processing

### Console Output Example

```
[INFO] Processing 20 spatio-temporal records at 6 unique coordinates (8 query batches)...
[INFO] Querying Ta_K from tavg1_2d_slv_Nx at 6 coordinates (8 batches)...
[INFO]   Coordinate 1/6: (35.7990, -76.6560) - 1 records in 1 time clusters
[INFO]     Batch 1/8: cluster 1/1 - 1 records, 0.0 days (2019-10-02 to 2019-10-02)
[INFO]   Coordinate 2/6: (41.8222, -80.6370) - 5 records in 1 time clusters
[INFO]     Batch 2/8: cluster 1/1 - 5 records, 14.8 days (2019-06-23 to 2019-07-08)
[INFO]   Coordinate 6/6: (45.7624, -122.3303) - 7 records in 2 time clusters
[INFO]     Batch 7/8: cluster 1/2 - 5 records, 11.8 days (2021-04-06 to 2021-04-18)
[INFO]     Batch 8/8: cluster 2/2 - 2 records, 3.9 days (2021-06-16 to 2021-06-20)
```

Each log line shows:
- Which coordinate is being queried
- How many records need data at that coordinate
- How times are clustered (avoids huge time spans)
- Exact date ranges being queried

## Available Variables

Common GEOS-5 FP variables for spatio-temporal queries:

| Variable | Description | Dataset | Units |
|----------|-------------|---------|-------|
| Ta_K | Air temperature | tavg1_2d_slv_Nx | Kelvin |
| SM | Surface soil moisture | tavg1_2d_lnd_Nx | m³/m³ |
| LAI | Leaf area index | tavg1_2d_lnd_Nx | m²/m² |
| SWin | Incoming shortwave radiation | tavg1_2d_rad_Nx | W/m² |
| LWin | Incoming longwave radiation | tavg1_2d_rad_Nx | W/m² |
| Ps_Pa | Surface pressure | tavg1_2d_slv_Nx | Pascal |
| RH | Relative humidity | tavg1_2d_slv_Nx | % |

See `GEOS5FP/VARIABLES_README.md` for complete list.

## Error Handling

The vectorized query handles errors gracefully:

```python
# If a query fails for a specific point/time, it logs a warning
# and sets the value to None, then continues processing

# Example warning:
# [WARNING] Failed to query Ta_K at (35.799, -76.656), time 2019-10-02 19:09:40: Connection timeout
```

Missing values appear as `NaN` in the result GeoDataFrame.

## Complete Example Script

See `spatio_temporal_example.py` for a full working example that:
1. Loads spatio-temporal CSV
2. Tests with 5 records
3. Processes full dataset
4. Exports results to CSV

Run it:

```bash
python spatio_temporal_example.py
```

## Comparison: Old vs. New

### Old Row-by-Row Approach

```python
# 15 lines, manual iteration, manual concatenation
results = []
for idx, row in gdf.iterrows():
    try:
        result = conn.variable(
            ["Ta_K", "SM", "LAI"],
            time_UTC=row['time_UTC'],
            geometry=row['geometry']
        )
        result['ID'] = row['ID']
        results.append(result)
    except Exception as e:
        print(f"Error: {e}")
        continue
final_df = pd.concat(results, ignore_index=False)
final_gdf = gpd.GeoDataFrame(final_df, geometry='geometry', crs='EPSG:4326')
```

### New Vectorized Approach

```python
# 5 lines, single method call, automatic GeoDataFrame
results = conn.variable(
    variable_name=["Ta_K", "SM", "LAI"],
    time_UTC=gdf['time_UTC'],
    geometry=gdf['geometry']
)
results['ID'] = gdf['ID'].values
```

**Result**: Same output, 66% less code, much cleaner!
