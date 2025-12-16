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
- `RH()` - Relative humidity
- `Ca()` / `CO2SC()` - Atmospheric CO2 concentration (ppmv)
- And many more (see [Available Variables](#available-variables) section below)

**Note:** For point queries and multi-variable queries, use the `.query()` method described in the next section.

### Generating Table Data with `.query()`

The `.query()` method is the recommended way to retrieve GEOS-5 FP data as tabular data (pandas DataFrames). It provides a flexible interface that supports:

- **Single point queries** - Data at one location and time
- **Time series queries** - Multiple timesteps at one location
- **Multi-variable queries** - Multiple variables in a single request
- **Vectorized spatio-temporal queries** - Multiple locations and times efficiently
- **Validation table generation** - Add GEOS-5 FP data to existing datasets

#### Single Point Query

```python
# Get data for single point at specific time
time_utc = datetime(2024, 11, 15, 12, 0)
lat, lon = 34.05, -118.25  # Los Angeles

result = conn.query(
    target_variables="Ta_K",
    time_UTC=time_utc,
    lat=lat,
    lon=lon
)
print(result)  # Returns DataFrame with temperature value
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

#### Validation Table Generation

```python
import geopandas as gpd
from shapely.geometry import Point

# Create a table with existing data
targets = gpd.GeoDataFrame({
    'time_UTC': [datetime(2024, 11, 15, 12), datetime(2024, 11, 15, 13)],
    'geometry': [Point(-118.25, 34.05), Point(-74.0, 40.7)],
    'site_name': ['Los Angeles', 'New York']
})

# Query variables and add as new columns to the table
result = conn.query(
    target_variables=["Ta_C", "RH", "SM"],
    targets_df=targets
)
print(result)  # Returns original table with new columns for each variable
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

You can query variables using either:
- **Predefined variable names** (e.g., `"Ta_K"`, `"SM"`, `"RH"`) - recommended for convenience
- **Raw GEOS-5 FP variable names** (e.g., `"T2M"`, `"SFMC"`, `"QV2M"`) - requires `dataset` parameter

```python
# Using predefined variable name (dataset automatically determined)
df = conn.query(
    target_variables="Ta_K",
    time_range=(start_time, end_time),
    lat=lat,
    lon=lon
)

# Using raw GEOS-5 FP variable name (dataset required)
df = conn.query(
    target_variables="T2M",  # Raw GEOS-5 FP variable name
    dataset="tavg1_2d_slv_Nx",  # Must specify product
    time_range=(start_time, end_time),
    lat=lat,
    lon=lon
)
```

See [Available Variables](#available-variables) section below for the complete list.

### Computed Variables

The package automatically computes derived meteorological variables from base GEOS-5 FP data. You can query these just like any other variable:

```python
# Query computed variables
results = conn.query(
    target_variables=["wind_speed_mps", "Ta_C", "RH", "VPD_kPa"],
    time_UTC=time_utc,
    lat=lat,
    lon=lon
)
```

Available computed variables:

| Variable | Description | Computed From | Units |
|----------|-------------|---------------|-------|
| `RH` | Relative humidity | Q, PS, Ta | fraction (0-1) |
| `Ta_C` | Air temperature in Celsius | Ta_K | °C |
| `wind_speed_mps` | Wind speed magnitude | U2M, V2M | m/s |
| `SVP_Pa` | Saturated vapor pressure | Ta | Pa |
| `Ea_Pa` | Actual vapor pressure | RH, SVP_Pa | Pa |
| `VPD_kPa` | Vapor pressure deficit | SVP_Pa, Ea_Pa | kPa |
| `Td_K` | Dew point temperature | Ta, RH | K |
| `PAR_proportion` | PAR albedo fraction | ALBVISDR, ALBEDO | fraction |
| `NIR_proportion` | NIR albedo fraction | ALBNIRDR, ALBEDO | fraction |

The package automatically retrieves only the necessary base variables and returns just the computed results.

## Available Variables

The package provides access to a comprehensive set of GEOS-5 FP variables, organized by category. All variables can be queried using the `.query()` method or their corresponding dedicated methods.

### Soil and Vegetation

| Variable | Description | Units | GEOS-5 FP Product |
|----------|-------------|-------|-------------------|
| `SM`, `SFMC` | Top layer soil moisture | fraction | tavg1_2d_lnd_Nx |
| `LAI` | Leaf area index | m²/m² | tavg1_2d_lnd_Nx |
| `LHLAND` | Latent heat flux over land | W/m² | tavg1_2d_lnd_Nx |
| `EFLUX` | Total latent energy flux | W/m² | tavg1_2d_flx_Nx |

### Temperature

| Variable | Description | Units | GEOS-5 FP Product |
|----------|-------------|-------|-------------------|
| `Ta`, `Ta_K` | Air temperature at 2m | K | tavg1_2d_slv_Nx |
| `Ta_C` | Air temperature in Celsius (computed) | °C | - |
| `Ts`, `Ts_K` | Surface temperature | K | tavg1_2d_slv_Nx |
| `Tmin`, `Tmin_K` | Minimum temperature at 2m | K | inst3_2d_asm_Nx |

### Pressure and Humidity

| Variable | Description | Units | GEOS-5 FP Product |
|----------|-------------|-------|-------------------|
| `PS` | Surface pressure | Pa | tavg1_2d_slv_Nx |
| `Q` | Specific humidity at 2m | kg/kg | tavg1_2d_slv_Nx |
| `RH` | Relative humidity (computed) | fraction | - |
| `SVP_Pa` | Saturated vapor pressure (computed) | Pa | - |
| `Ea_Pa` | Actual vapor pressure (computed) | Pa | - |
| `VPD_kPa` | Vapor pressure deficit (computed) | kPa | - |
| `Td_K` | Dew point temperature (computed) | K | - |

### Wind

| Variable | Description | Units | GEOS-5 FP Product |
|----------|-------------|-------|-------------------|
| `U2M` | Eastward wind at 2m | m/s | inst3_2d_asm_Nx |
| `V2M` | Northward wind at 2m | m/s | inst3_2d_asm_Nx |
| `wind_speed_mps` | Wind speed magnitude (computed) | m/s | - |

### Atmospheric Composition

| Variable | Description | Units | GEOS-5 FP Product |
|----------|-------------|-------|-------------------|
| `vapor_kgsqm`, `vapor_gccm` | Total precipitable water vapor | kg/m² | inst3_2d_asm_Nx |
| `ozone_dobson`, `ozone_cm` | Total column ozone | Dobson units / cm-atm | inst3_2d_asm_Nx |
| `Ca`, `CO2SC` | Atmospheric CO2 concentration | ppmv | tavg3_2d_chm_Nx |
| `AOT` | Aerosol optical thickness | dimensionless | tavg3_2d_aer_Nx |
| `COT` | Cloud optical thickness | dimensionless | tavg1_2d_rad_Nx |

### Radiation

| Variable | Description | Units | GEOS-5 FP Product |
|----------|-------------|-------|-------------------|
| `SWin` | Net surface shortwave radiation | W/m² | tavg1_2d_rad_Nx |
| `SWTDN` | Surface incident shortwave flux | W/m² | tavg1_2d_rad_Nx |
| `PARDR` | Direct photosynthetically active radiation | W/m² | tavg1_2d_lnd_Nx |
| `PARDF` | Diffuse photosynthetically active radiation | W/m² | tavg1_2d_lnd_Nx |

### Albedo

| Variable | Description | Units | GEOS-5 FP Product |
|----------|-------------|-------|-------------------|
| `ALBEDO` | Total surface albedo | fraction | tavg1_2d_rad_Nx |
| `ALBVISDR` | Direct beam visible surface albedo | fraction | tavg1_2d_rad_Nx |
| `ALBVISDF` | Diffuse beam visible surface albedo | fraction | tavg1_2d_rad_Nx |
| `ALBNIRDR` | Direct beam NIR surface albedo | fraction | tavg1_2d_rad_Nx |
| `ALBNIRDF` | Diffuse beam NIR surface albedo | fraction | tavg1_2d_rad_Nx |
| `PAR_proportion` | PAR albedo fraction (computed) | fraction | - |
| `NIR_proportion` | NIR albedo fraction (computed) | fraction | - |

### Variable Aliases

Many variables have multiple names for convenience:

- `SM` ↔ `SFMC` (Soil moisture)
- `Ta` ↔ `Ta_K` (Air temperature)
- `Ts` ↔ `Ts_K` (Surface temperature)
- `Tmin` ↔ `Tmin_K` (Minimum temperature)
- `Ca` ↔ `CO2SC` (CO2 concentration)
- `vapor_kgsqm` ↔ `vapor_gccm` (Water vapor)
- `ozone_dobson` ↔ `ozone_cm` (Ozone)

### Temporal Resolution

- **Hourly** (`tavg1_*`): 1-hour time-averaged data
- **3-Hourly** (`tavg3_*`, `inst3_*`): 3-hour time-averaged or instantaneous data

### Complete Variable Reference

For detailed information about all variables, including formulas for computed variables and usage examples, see:
- **[QUERYABLE_VARIABLES.md](QUERYABLE_VARIABLES.md)** - Comprehensive variable guide
- **[GEOS5FP/variables.csv](GEOS5FP/variables.csv)** - Machine-readable variable mappings

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
