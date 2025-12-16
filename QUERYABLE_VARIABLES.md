# GEOS-5 FP Queryable Variables Guide

This document describes all variables that can be queried using the `.query()` method in the GEOS-5 FP Python package.

## Table of Contents

- [Overview](#overview)
- [Basic Variable Categories](#basic-variable-categories)
- [How to Query Variables](#how-to-query-variables)
- [Direct GEOS-5 FP Variables](#direct-geos-5-fp-variables)
- [Computed/Derived Variables](#computedderived-variables)
- [Variable Aliases](#variable-aliases)

## Overview

The GEOS-5 FP package provides access to NASA's GEOS-5 Forward Processing meteorological data through two types of variables:

1. **Direct Variables**: Retrieved directly from GEOS-5 FP NetCDF files
2. **Computed Variables**: Derived from base GEOS-5 FP variables using physical equations

You can query variables using:
- Predefined variable names (e.g., `"Ta_K"`, `"SM"`, `"RH"`)
- Raw GEOS-5 FP variable names (e.g., `"T2M"`, `"SFMC"`, `"QV2M"`)
- Variable aliases for convenience

## Basic Variable Categories

Variables are organized by the GEOS-5 FP product they come from:

- **Meteorology**: Temperature, pressure, humidity, wind
- **Land Surface**: Soil moisture, vegetation, surface fluxes
- **Radiation**: Incoming/outgoing radiation, albedo components
- **Atmospheric Composition**: Aerosols, ozone, CO2
- **Computed**: Derived quantities like relative humidity, vapor pressure deficit

## How to Query Variables

### Single Variable Query

```python
from GEOS5FP import GEOS5FPConnection
from datetime import datetime

conn = GEOS5FPConnection()

# Query air temperature
result = conn.query(
    "Ta_K",
    time_UTC=datetime(2024, 11, 15, 12),
    lat=34.05,
    lon=-118.25
)
```

### Multiple Variable Query

```python
# Query multiple variables at once
result = conn.query(
    ["Ta_K", "RH", "SM", "SWin"],
    time_UTC=datetime(2024, 11, 15, 12),
    lat=34.05,
    lon=-118.25
)
```

### Using Raw GEOS-5 FP Variable Names

```python
# Query using raw variable names (requires dataset parameter)
result = conn.query(
    "T2M",
    dataset="tavg1_2d_slv_Nx",
    time_UTC=datetime(2024, 11, 15, 12),
    lat=34.05,
    lon=-118.25
)
```

## Direct GEOS-5 FP Variables

These variables are retrieved directly from GEOS-5 FP products without computation.

### Soil and Vegetation

| Variable Name | Description | Product | Units | Raw Variable |
|--------------|-------------|---------|-------|--------------|
| `SM` | Top layer soil moisture | `tavg1_2d_lnd_Nx` | fraction | `SFMC` |
| `SFMC` | Top layer soil moisture | `tavg1_2d_lnd_Nx` | fraction | `SFMC` |
| `LAI` | Leaf area index | `tavg1_2d_lnd_Nx` | m²/m² | `LAI` |
| `LHLAND` | Latent heat flux over land | `tavg1_2d_lnd_Nx` | W/m² | `LHLAND` |
| `EFLUX` | Total latent energy flux | `tavg1_2d_flx_Nx` | W/m² | `EFLUX` |

### Temperature

| Variable Name | Description | Product | Units | Raw Variable |
|--------------|-------------|---------|-------|--------------|
| `Ts` | Surface temperature | `tavg1_2d_slv_Nx` | K | `TS` |
| `Ts_K` | Surface temperature | `tavg1_2d_slv_Nx` | K | `TS` |
| `Ta` | Air temperature at 2m | `tavg1_2d_slv_Nx` | K | `T2M` |
| `Ta_K` | Air temperature at 2m | `tavg1_2d_slv_Nx` | K | `T2M` |
| `Tmin` | Minimum temperature at 2m | `inst3_2d_asm_Nx` | K | `T2MMIN` |
| `Tmin_K` | Minimum temperature at 2m | `inst3_2d_asm_Nx` | K | `T2MMIN` |

### Pressure and Humidity

| Variable Name | Description | Product | Units | Raw Variable |
|--------------|-------------|---------|-------|--------------|
| `PS` | Surface pressure | `tavg1_2d_slv_Nx` | Pa | `PS` |
| `Q` | Specific humidity at 2m | `tavg1_2d_slv_Nx` | kg/kg | `QV2M` |

### Wind

| Variable Name | Description | Product | Units | Raw Variable |
|--------------|-------------|---------|-------|--------------|
| `U2M` | Eastward wind at 2m | `inst3_2d_asm_Nx` | m/s | `U2M` |
| `V2M` | Northward wind at 2m | `inst3_2d_asm_Nx` | m/s | `V2M` |

### Atmospheric Composition

| Variable Name | Description | Product | Units | Raw Variable |
|--------------|-------------|---------|-------|--------------|
| `vapor_kgsqm` | Total precipitable water vapor | `inst3_2d_asm_Nx` | kg/m² | `TQV` |
| `vapor_gccm` | Total precipitable water vapor | `inst3_2d_asm_Nx` | kg/m² | `TQV` |
| `ozone_dobson` | Total column ozone | `inst3_2d_asm_Nx` | Dobson units | `TO3` |
| `ozone_cm` | Total column ozone | `inst3_2d_asm_Nx` | cm-atm | `TO3` |
| `CO2SC` | Atmospheric CO2 concentration | `tavg3_2d_chm_Nx` | ppmv | `CO2SC` |
| `Ca` | Atmospheric CO2 concentration | `tavg3_2d_chm_Nx` | ppmv | `CO2SC` |

### Radiation

| Variable Name | Description | Product | Units | Raw Variable |
|--------------|-------------|---------|-------|--------------|
| `SWin` | Net surface shortwave radiation | `tavg1_2d_rad_Nx` | W/m² | `SWGNT` |
| `SWTDN` | Surface incident shortwave flux | `tavg1_2d_rad_Nx` | W/m² | `SWTDN` |

### Photosynthetically Active Radiation (PAR)

| Variable Name | Description | Product | Units | Raw Variable |
|--------------|-------------|---------|-------|--------------|
| `PARDR` | Direct photosynthetically active radiation | `tavg1_2d_lnd_Nx` | W/m² | `PARDR` |
| `PARDF` | Diffuse photosynthetically active radiation | `tavg1_2d_lnd_Nx` | W/m² | `PARDF` |

### Albedo Components

| Variable Name | Description | Product | Units | Raw Variable |
|--------------|-------------|---------|-------|--------------|
| `ALBVISDR` | Direct beam visible surface albedo | `tavg1_2d_rad_Nx` | fraction | `ALBVISDR` |
| `ALBVISDF` | Diffuse beam visible surface albedo | `tavg1_2d_rad_Nx` | fraction | `ALBVISDF` |
| `ALBNIRDR` | Direct beam NIR surface albedo | `tavg1_2d_rad_Nx` | fraction | `ALBNIRDR` |
| `ALBNIRDF` | Diffuse beam NIR surface albedo | `tavg1_2d_rad_Nx` | fraction | `ALBNIRDF` |
| `ALBEDO` | Total surface albedo | `tavg1_2d_rad_Nx` | fraction | `ALBEDO` |

### Aerosols and Clouds

| Variable Name | Description | Product | Units | Raw Variable |
|--------------|-------------|---------|-------|--------------|
| `AOT` | Aerosol optical thickness | `tavg3_2d_aer_Nx` | dimensionless | `TOTEXTTAU` |
| `COT` | Cloud optical thickness | `tavg1_2d_rad_Nx` | dimensionless | `TAUTOT` |

## Computed/Derived Variables

These variables are automatically calculated from base GEOS-5 FP variables. When you query them, the package:

1. Identifies required base variables
2. Queries them from appropriate GEOS-5 FP products
3. Computes the derived variable
4. Returns only the requested variable

### Humidity Variables

| Variable Name | Description | Formula | Units |
|--------------|-------------|---------|-------|
| `RH` | Relative humidity | Computed from `Q`, `PS`, `Ta` | fraction (0-1) |
| `SVP_Pa` | Saturated vapor pressure | Computed from `Ta` using Clausius-Clapeyron | Pa |
| `Ea_Pa` | Actual vapor pressure | `RH × SVP_Pa` | Pa |
| `VPD_kPa` | Vapor pressure deficit | `(SVP_Pa - Ea_Pa) / 1000` | kPa |
| `Td_K` | Dew point temperature | Computed from `Ta` and `RH` | K |

### Temperature Variables

| Variable Name | Description | Formula | Units |
|--------------|-------------|---------|-------|
| `Ta_C` | Air temperature in Celsius | `Ta_K - 273.15` | °C |

### Wind Variables

| Variable Name | Description | Formula | Units |
|--------------|-------------|---------|-------|
| `wind_speed_mps` | Wind speed magnitude | `√(U2M² + V2M²)` | m/s |

### Albedo Proportions

| Variable Name | Description | Formula | Units |
|--------------|-------------|---------|-------|
| `PAR_proportion` | PAR albedo fraction | `ALBVISDR / ALBEDO` | fraction |
| `NIR_proportion` | NIR albedo fraction | `ALBNIRDR / ALBEDO` | fraction |

**Usage Example:**

```python
# These proportions can be used as scaling factors
# to derive component albedos from total albedo
result = conn.query(
    ["ALBEDO", "PAR_proportion"],
    time_UTC=datetime(2024, 11, 15, 12),
    lat=34.05,
    lon=-118.25
)

# Calculate visible albedo
albedo_PAR = result['ALBEDO'] * result['PAR_proportion']
```

## Variable Aliases

Many variables have multiple names for convenience. All aliases point to the same underlying data:

| Primary Name | Aliases | Description |
|-------------|---------|-------------|
| `SFMC` | `SM` | Soil moisture |
| `T2M` (raw) | `Ta`, `Ta_K` | Air temperature |
| `TS` (raw) | `Ts`, `Ts_K` | Surface temperature |
| `T2MMIN` (raw) | `Tmin`, `Tmin_K` | Minimum temperature |
| `TQV` (raw) | `vapor_kgsqm`, `vapor_gccm` | Water vapor |
| `TO3` (raw) | `ozone_dobson`, `ozone_cm` | Ozone |
| `CO2SC` | `Ca` | Atmospheric CO2 |

## Temporal Resolution

Different GEOS-5 FP products have different temporal resolutions:

- **Hourly** (`tavg1_*`): 1-hour time-averaged data
- **3-Hourly** (`tavg3_*`, `inst3_*`): 3-hour time-averaged or instantaneous data

When querying variables from different products simultaneously, the package handles temporal alignment automatically using the `temporal_interpolation` parameter (default: `"interpolate"`).

## Spatial Resolution

All GEOS-5 FP products have native spatial resolution of:
- **Longitude**: 0.625° (~69 km at equator)
- **Latitude**: 0.5° (~55 km)

The `.query()` method can automatically resample to any target geometry using the `geometry` parameter.

## Additional Notes

### Point Queries vs Raster Queries

- **Point queries** (using `lat`/`lon` or `Point` geometry): Support multiple variables simultaneously
- **Raster queries** (using `RasterGeometry`): Currently support single variables only

### OPeNDAP Support

Point queries use NASA's OPeNDAP service for efficient time-series retrieval. This requires:
```bash
conda install -c conda-forge xarray netcdf4
```

### Variable Discovery

To see all available variables programmatically:

```python
from GEOS5FP.constants import GEOS5FP_VARIABLES, COMPUTED_VARIABLES

# Direct variables
print("Direct variables:", GEOS5FP_VARIABLES.keys())

# Computed variables
print("Computed variables:", COMPUTED_VARIABLES)
```

## Examples

### Example 1: Multi-Variable Time Series

```python
from datetime import datetime, timedelta
from GEOS5FP import GEOS5FPConnection

conn = GEOS5FPConnection()
end_time = datetime(2024, 11, 15)
start_time = end_time - timedelta(days=7)

# Query multiple variables for a week-long time series
df = conn.query(
    target_variables=["Ta_C", "RH", "VPD_kPa", "wind_speed_mps", "SM"],
    time_range=(start_time, end_time),
    lat=34.05,
    lon=-118.25
)

print(df)
```

### Example 2: Validation Dataset

```python
import geopandas as gpd
from shapely.geometry import Point

# Create validation points
targets = gpd.GeoDataFrame({
    'time_UTC': [datetime(2024, 11, 15, 12), datetime(2024, 11, 15, 13)],
    'geometry': [Point(-118.25, 34.05), Point(-74.0, 40.7)],
    'site_name': ['Los Angeles', 'New York']
})

# Query variables and add as columns
result = conn.query(
    target_variables=["Ta_C", "RH", "SM", "SWin"],
    targets_df=targets
)

print(result)
```

### Example 3: Albedo Components

```python
# Query all albedo components
result = conn.query(
    target_variables=[
        "ALBEDO",
        "ALBVISDR", "ALBVISDF",
        "ALBNIRDR", "ALBNIRDF",
        "PAR_proportion", "NIR_proportion"
    ],
    time_UTC=datetime(2024, 11, 15, 12),
    lat=34.05,
    lon=-118.25
)

print(result)
```

## Support and Documentation

For more information:
- **Multi-Variable Guide**: See `guides/Multi Variable Guide.md`
- **Download Guide**: See `guides/Download and Validation Guide.md`
- **Variable Definitions**: See `GEOS5FP/variables.csv`
- **GitHub Repository**: [JPL-Evapotranspiration-Algorithms/GEOS5FP](https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP)
