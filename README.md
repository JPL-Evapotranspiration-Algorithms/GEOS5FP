# `GEOS5FP` Python Package

[![CI](https://github.com/JPL-Evapotranspiration-Algorithms/geos5fp/actions/workflows/ci.yml/badge.svg)](https://github.com/JPL-Evapotranspiration-Algorithms/geos5fp/actions/workflows/ci.yml)

The `GEOS5FP` Python package generates rasters of near-real-time GEOS-5 FP near-surface meteorology.

[Gregory H. Halverson](https://github.com/gregory-halverson-jpl) (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

## Installation

This package is available on PyPi as a [pip package](https://pypi.org/project/geos5fp/) called `GEOS5FP`.

```bash
pip install GEOS5FP
```

## Usage

Import this package as `GEOS5FP`.

```python
from GEOS5FP import GEOS5FPConnection
from datetime import datetime
```

### Creating a Connection

```python
# Create connection to GEOS-5 FP data
conn = GEOS5FPConnection()
```

### Generating Raster Data

Generate georeferenced raster data for a specific time and optional target geometry:

```python
from rasters import RasterGeometry

# Define target geometry (optional - if not provided, uses native GEOS-5 FP grid)
target_geometry = RasterGeometry.open("target_area.tif")

# Get air temperature raster for a specific time
time_utc = datetime(2024, 11, 15, 12, 0)
temperature_raster = conn.Ta_K(time_UTC=time_utc, geometry=target_geometry)

# Get soil moisture raster
soil_moisture_raster = conn.SM(time_UTC=time_utc, geometry=target_geometry)

# Get leaf area index raster
lai_raster = conn.LAI(time_UTC=time_utc, geometry=target_geometry)

# Save raster to file
temperature_raster.to_geotiff("temperature.tif")
```

Available raster methods include:
- `Ta_K()` - Air temperature (Kelvin)
- `Ts_K()` - Surface temperature (Kelvin)
- `SM()` / `SFMC()` - Soil moisture
- `LAI()` - Leaf area index
- `NDVI()` - Normalized difference vegetation index
- `RH()` - Relative humidity
- `Ca()` / `CO2SC()` - Atmospheric CO2 concentration (ppmv)
- And many more (see `variables.csv` for complete list)

### Generating Table Data

Query point locations or time series to generate tabular data as pandas DataFrames:

#### Single Point Query

```python
from shapely.geometry import Point

# Define point location (longitude, latitude)
point = Point(-118.25, 34.05)  # Los Angeles

# Get data for single point at specific time
time_utc = datetime(2024, 11, 15, 12, 0)
result = conn.Ta_K(time_UTC=time_utc, geometry=point)
print(result)  # Returns DataFrame with temperature value
```

#### Multiple Points Query

```python
from shapely.geometry import MultiPoint

# Define multiple points
points = MultiPoint([
    (-118.25, 34.05),   # Los Angeles
    (-122.42, 37.77),   # San Francisco
    (-73.94, 40.73)     # New York
])

# Query multiple points at once
results = conn.Ta_K(time_UTC=time_utc, geometry=points)
print(results)  # Returns DataFrame with one row per point
```

#### Time Series Query

```python
from datetime import timedelta

# Define time range
end_time = datetime(2024, 11, 15, 0, 0)
start_time = end_time - timedelta(days=7)  # 7 days of data

# Get time series for a point location
lat, lon = 34.05, -118.25
df = conn.query(
    target_variables="Ta_K",
    time_range=(start_time, end_time),
    lat=lat,
    lon=lon
)
print(df)  # Returns DataFrame with time series
```

#### Multi-Variable Query

```python
# Query multiple variables at once
variables = ["Ta_K", "SM", "LAI"]
df_multi = conn.query(
    target_variables=variables,
    time_range=(start_time, end_time),
    lat=lat,
    lon=lon
)
print(df_multi)  # Returns DataFrame with columns for each variable
```

#### Vectorized Spatio-Temporal Query

```python
import pandas as pd
import geopandas as gpd

# Load spatio-temporal data from CSV
data = pd.read_csv("locations.csv")  # Should have columns: time_UTC, lat, lon
data['time_UTC'] = pd.to_datetime(data['time_UTC'])

# Create geometries
gdf = gpd.GeoDataFrame(
    data,
    geometry=gpd.points_from_xy(data['lon'], data['lat'])
)

# Query all points and times at once (vectorized operation)
results = conn.query(
    target_variables=["Ta_K", "SM", "LAI"],
    time_UTC=gdf['time_UTC'],
    geometry=gdf['geometry']
)
print(results)  # Returns DataFrame with results for all locations and times
```

### Using Raw GEOS-5 FP Variables

You can also query variables directly by their GEOS-5 FP product and variable names:

```python
# Query specific humidity from tavg1_2d_slv_Nx product
df = conn.query(
    target_variables="QV2M",  # Raw GEOS-5 FP variable name
    time_range=(start_time, end_time),
    dataset="tavg1_2d_slv_Nx",
    lat=lat,
    lon=lon
)
```

See `GEOS5FP/variables.csv` for the complete list of available variables and their mappings.

### Computed Variables

The package automatically computes derived meteorological variables from base GEOS-5 FP data. You can query these just like any other variable:

```python
# Query computed variables
results = conn.query(
    target_variables=["wind_speed_mps", "Ta_C", "RH"],
    time_UTC=time_utc,
    lat=lat,
    lon=lon
)
```

Available computed variables:
- **`wind_speed_mps`** - Wind speed in m/s (from U2M and V2M components)
- **`Ta_C`** - Air temperature in Celsius (from Ta_K)
- **`RH`** - Relative humidity (from Q, PS, Ta)
- **`VPD_kPa`** - Vapor pressure deficit in kPa (from SVP and Ea)
- **`Ea_Pa`** - Actual vapor pressure in Pascals
- **`SVP_Pa`** - Saturated vapor pressure in Pascals
- **`Td_K`** - Dew point temperature in Kelvin

The package automatically retrieves only the necessary base variables and returns just the computed results.

## Data Source & Citation

This package accesses GEOS-5 FP (Forward Processing) data produced by the Global Modeling and Assimilation Office (GMAO) at NASA Goddard Space Flight Center.

### Data Access

GEOS-5 FP data is accessed through:
- **OPeNDAP Server**: `https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/`
- **HTTP Server**: `https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das`

Data is provided by NASA's Center for Climate Simulation (NCCS).

### Citation

When using GEOS-5 FP data in publications, please cite:

**Data Product:**
```
Global Modeling and Assimilation Office (GMAO) (2015), GEOS-5 FP: GEOS Forward 
Processing for Instrument Support, Greenbelt, MD, USA, Goddard Earth Sciences 
Data and Information Services Center (GES DISC). 
Accessed: [Date]
```

**Acknowledgment:**
```
GEOS-5 FP data used in this study were provided by the Global Modeling and 
Assimilation Office (GMAO) at NASA Goddard Space Flight Center through the 
NASA Center for Climate Simulation (NCCS).
```

For more information about GEOS-5 FP, visit: https://gmao.gsfc.nasa.gov/GEOS/
