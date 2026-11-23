# `GEOS5FP` Python Package

[![CI](https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP/actions/workflows/ci.yml/badge.svg)](https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP/actions/workflows/ci.yml)

The `GEOS5FP` Python package provides programmatic retrieval, spatio-temporal interpolation, empirical downscaling, and tabular extraction for NASA's near-real-time **GEOS-5 Forward Processing (GEOS-5 FP)** meteorological assimilation products.

[Gregory H. Halverson](https://github.com/gregory-halverson-jpl) (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

## Citation

If you use the `GEOS5FP` Python package in your research or software, please cite it as:

```text
Halverson, G. H. (2026). GEOS5FP: Near-real-time GEOS-5 FP near-surface meteorology in Python (Version 2.25.0) [Computer software]. NASA Jet Propulsion Laboratory. https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP
```

**BibTeX:**

```bibtex
@software{Halverson_GEOS5FP_2026,
  author       = {Halverson, Gregory H.},
  title        = {{GEOS5FP}: Near-real-time {GEOS-5 FP} near-surface meteorology in {Python}},
  year         = {2026},
  version      = {2.25.0},
  institution  = {NASA Jet Propulsion Laboratory},
  url          = {https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP}
}
```

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Creating a Connection](#creating-a-connection)
  - [Downloading, Staging & NetCDF Validation](#downloading-staging--netcdf-validation)
  - [Generating Gridded Raster Data](#generating-gridded-raster-data)
  - [Generating Tabular & Point Data with `.query()`](#generating-tabular--point-data-with-query)
  - [Using Raw GEOS-5 FP Variable Names](#using-raw-geos-5-fp-variable-names)
  - [Querying Computed Variables](#querying-computed-variables)
  - [Spatial Downscaling](#spatial-downscaling)
- [Complete Queryable Variables Reference](#complete-queryable-variables-reference)
  - [Direct GEOS-5 FP Variables](#direct-geos-5-fp-variables)
  - [Computed / Derived Variables](#computed--derived-variables)
  - [Variable Aliases](#variable-aliases)
  - [Spatial & Temporal Resolution](#spatial--temporal-resolution)
  - [Programmatic Variable Discovery](#programmatic-variable-discovery)
- [Known Issues & Operational Notes](#known-issues--operational-notes)
- [Release History & Technical Evolution](#release-history--technical-evolution)
- [Data Source & Data Product Citation](#data-source--data-product-citation)

## Installation

`GEOS5FP` requires Python `>=3.11` and is available on PyPI:

```bash
pip install GEOS5FP
```

To install optional dependencies used by the scripts in `examples/` (such as `ECOv002-calval-tables` and `sentinel-tiles`):

```bash
pip install "GEOS5FP[examples]"
```

For local development and running the test suite:

```bash
git clone https://github.com/JPL-Evapotranspiration-Algorithms/GEOS5FP.git
cd GEOS5FP
pip install -e ".[dev]"
pytest
```

## Usage

Import `GEOS5FP` (alias for `GEOS5FPConnection`) or `GEOS5FPConnection` directly:

```python
from datetime import datetime, timedelta
from GEOS5FP import GEOS5FP, GEOS5FPConnection
```

### Creating a Connection

```python
conn = GEOS5FPConnection(
    download_directory="~/data/GEOS5FP",  # Final destination for downloaded .nc4 granules
    remote="https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das",
    verbose=True,
    use_opendap=True,         # Use OPeNDAP for fast point/table queries
    allow_fallback=False,     # Set True to fall back to HTTP granule downloads if OPeNDAP fails
    listing_ttl_seconds=600,  # Cache remote directory listings for 10 minutes
)
```

### Downloading, Staging & NetCDF Validation

`GEOS5FPConnection.download_file()` downloads, validates, and caches GEOS-5 FP NetCDF4 granules:

```python
conn = GEOS5FPConnection(download_directory="~/data/GEOS5FP")
granule = conn.download_file(URL=granule_url)
print(granule.filename)
```

How granule downloads and validation work:
- **Out-of-tree Pooch staging:** Downloads use `pooch.HTTPDownloader` (1 MiB chunks, 120s timeout, verified HTTPS). Each download attempt stages network chunks inside a fresh temporary directory under `pooch.os_cache("GEOS5FP")`, outside the configured `download_directory`—preventing partial-file lock or sync timeouts when `~/data` resides on cloud-backed storage (such as OneDrive or Google Drive).
- **Atomic publication & inter-process locking:** Once the network transfer finishes, the completed file is published to `download_directory` via atomic `os.replace` (with a cross-filesystem temporary copy fallback) under an exclusive inter-process file lock (`_granule_file_lock`).
- **Crash-isolated NetCDF4 validation:** Every existing or newly downloaded `.nc4` granule is validated via `validate_GEOS5FP_NetCDF_file()` using `netCDF4.Dataset` inside an isolated subprocess (`use_subprocess=True` by default). This protects parent Python workflows from low-level C++ `SIGABRT` or segmentation faults in HDF5/GDAL libraries if a truncated file is encountered.

You can also validate files or directories directly:

```python
from GEOS5FP import validate_GEOS5FP_NetCDF_file, validate_GEOS5FP_directory, is_valid_GEOS5FP_file

result = validate_GEOS5FP_NetCDF_file("GEOS.fp.asm.tavg1_2d_slv_Nx.20250214_1200.V01.nc4")
print(result.is_valid, result.metadata)
```

### Generating Gridded Raster Data

Retrieve georeferenced `rasters.Raster` objects at any UTC timestamp, optionally reprojected/resampled to a target `RasterGeometry` or `RasterGrid` (default resampling: `"lanczos"` for gridded targets, `"nearest"` for points):

```python
from rasters import RasterGeometry

# Define target geometry (optional; omit for native 0.3125° x 0.25° GEOS-5 FP global grid)
target_geometry = RasterGeometry.open("target_area.tif")
time_utc = datetime(2024, 11, 15, 12, 0)

# Single-variable dedicated methods
Ta_K = conn.Ta_K(time_UTC=time_utc, geometry=target_geometry)
Ta_C = conn.Ta_C(time_UTC=time_utc, geometry=target_geometry)
RH = conn.RH(time_UTC=time_utc, geometry=target_geometry)
SM = conn.SM(time_UTC=time_utc, geometry=target_geometry)
LAI = conn.LAI(time_UTC=time_utc, geometry=target_geometry)
Ca = conn.Ca(time_UTC=time_utc, geometry=target_geometry)

# Save raster to GeoTIFF
Ta_C.to_geotiff("air_temperature_C.tif")
```

You can also query gridded rasters through `.query()` using a scalar variable name, a 1-element list, or a multi-variable list (which returns a `dict[str, Raster]`):

```python
# Single-element list on a gridded geometry returns a Raster
co2_raster = conn.query(variables=["CO2SC"], time_UTC=time_utc, geometry=target_geometry)

# Multi-variable list on a gridded geometry returns a dict mapping variable names to Rasters
raster_dict = conn.query(
    target_variables=["Ta_C", "RH", "SM"],
    time_UTC=time_utc,
    geometry=target_geometry,
)
```

### Generating Tabular & Point Data with `.query()`

The `.query()` method (available both as `conn.query(...)` and as a standalone function `GEOS5FP.query(...)`) provides a unified interface across single points, `MultiPoint` collections, time series, and spatio-temporal tables.

#### 1. Single Point and `MultiPoint` Queries

Single-variable point queries return a 1D `numpy.ndarray`; multi-variable point queries return a `geopandas.GeoDataFrame`:

```python
from shapely.geometry import Point, MultiPoint

time_utc = datetime(2024, 11, 15, 12, 0)

# Single variable at a point -> returns 1D np.ndarray
ta_val = conn.query(
    target_variables="Ta_K",
    time_UTC=time_utc,
    lat=34.05,
    lon=-118.25,
)

# Multiple points at a single timestamp -> returns 1D np.ndarray of length 3
cities = MultiPoint([
    (-118.25, 34.05),   # Los Angeles
    (-122.42, 37.77),   # San Francisco
    (-73.94, 40.73),    # New York
])
aot_vals = conn.AOT(time_UTC=time_utc, geometry=cities)

# Multiple variables at a point -> returns GeoDataFrame
point_gdf = conn.query(
    target_variables=["Ta_C", "RH", "VPD_kPa", "wind_speed_mps"],
    time_UTC=time_utc,
    geometry=Point(-118.25, 34.05),
)
```

#### 2. Point Time-Series Queries (`time_range`)

Pass `time_range=(start_time, end_time)` to query OPeNDAP time slices efficiently (automatically chunked into 7-day windows for long single-point requests):

```python
end_time = datetime(2024, 11, 15, 0, 0)
start_time = end_time - timedelta(days=7)

# Single variable time-series -> returns 1D np.ndarray
ta_series = conn.query(
    target_variables="Ta_K",
    time_range=(start_time, end_time),
    lat=34.05,
    lon=-118.25,
)

# Multi-variable time-series -> returns time-indexed GeoDataFrame
df_ts = conn.query(
    target_variables=["Ta_C", "RH", "SM", "SWin"],
    time_range=(start_time, end_time),
    lat=34.05,
    lon=-118.25,
    temporal_interpolation="interpolate",  # or "nearest"
)
print(df_ts.head())
```

#### 3. Enriching Existing Tables (`targets_df`)

Pass a `DataFrame` or `GeoDataFrame` containing `time_UTC` and `geometry` columns via `targets_df` to append queried variables as new columns:

```python
import geopandas as gpd
from shapely.geometry import Point

targets = gpd.GeoDataFrame(
    {
        "site_name": ["Los Angeles", "New York"],
        "time_UTC": [datetime(2024, 11, 15, 12), datetime(2024, 11, 15, 13)],
        "geometry": [Point(-118.25, 34.05), Point(-74.0, 40.7)],
    },
    crs="EPSG:4326",
)

enriched = conn.query(
    target_variables=["Ta_C", "RH", "SM", "COT", "AOT", "vapor_gccm", "ozone_cm"],
    targets_df=targets,
)
print(enriched)
```

#### 4. Vectorized Spatio-Temporal Batch Queries

Pass aligned sequences (`list`, `tuple`, `np.ndarray`, `pd.Series`, or `gpd.GeoSeries`) of timestamps and geometries. `GEOS5FP` automatically groups records by unique `(lat, lon)` coordinates, clusters timestamps into 1-day batches, fetches all variables per dataset in consolidated OPeNDAP calls, and linearly interpolates to each target timestamp:

```python
import pandas as pd
import geopandas as gpd

data = pd.read_csv("locations.csv")  # columns: time_UTC, lat, lon
data["time_UTC"] = pd.to_datetime(data["time_UTC"])
gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data["lon"], data["lat"]), crs="EPSG:4326")

results = conn.query(
    target_variables=["Ta_K", "RH", "SM", "LAI"],
    time_UTC=gdf["time_UTC"],
    geometry=gdf["geometry"],
)
print(results)
```

### Using Raw GEOS-5 FP Variable Names

In addition to predefined variable names, you can query any raw GEOS-5 FP variable directly by specifying `dataset`:

```python
df_raw = conn.query(
    target_variables=["T2M", "PS", "QV2M"],
    dataset="tavg1_2d_slv_Nx",
    time_range=(start_time, end_time),
    lat=34.05,
    lon=-118.25,
)
```

### Querying Computed Variables

All 14 derived meteorological and radiative variables in `COMPUTED_VARIABLES` are supported across gridded raster, single-point, multi-point, time-series, and vectorized batch queries. Required base variables are resolved and fetched automatically and pruned from the final output:

```python
results = conn.query(
    target_variables=[
        "Ta_C",
        "Tmin_C",
        "RH",
        "SVP_Pa",
        "Ea_Pa",
        "VPD_kPa",
        "Td_K",
        "wind_speed_mps",
        "PAR_proportion",
        "NIR_proportion",
        "PAR_albedo",
        "NIR_albedo",
        "albedo_visible",
        "albedo_NIR",
    ],
    time_UTC=datetime(2024, 11, 15, 12, 0),
    lat=34.05,
    lon=-118.25,
)
```

### Spatial Downscaling

`GEOS5FP.downscaling` provides empirical thermal and moisture downscaling routines (`linear_downscale`, `bias_correct`, `downscale_air_temperature`, `downscale_soil_moisture`, `downscale_vapor_pressure_deficit`, and `downscale_relative_humidity`), and `conn.Ta_K()` accepts a fine-resolution surface temperature raster (`ST_K`) and optional `water` mask directly:

```python
# Downscale coarse GEOS-5 FP air temperature using fine-resolution surface temperature (ST_K)
Ta_K_fine = conn.Ta_K(
    time_UTC=datetime(2024, 11, 15, 12, 0),
    ST_K=land_surface_temp_raster,
    water=water_mask_raster,
)
```

## Complete Queryable Variables Reference

Variables are defined in `GEOS5FP/variables.csv` (direct GEOS-5 FP variables) and `GEOS5FP/constants.py` (`COMPUTED_VARIABLES`).

### Direct GEOS-5 FP Variables

#### Soil and Vegetation

| Variable Name | Description | GEOS-5 FP Product | Raw Variable | Units |
| :--- | :--- | :--- | :--- | :--- |
| `SM`, `SFMC` | Top layer soil moisture | `tavg1_2d_lnd_Nx` | `SFMC` | fraction ($\text{m}^3\,\text{m}^{-3}$) |
| `LAI` | Leaf area index | `tavg1_2d_lnd_Nx` | `LAI` | $\text{m}^2\,\text{m}^{-2}$ |
| `LHLAND` | Latent heat flux over land | `tavg1_2d_lnd_Nx` | `LHLAND` | $\text{W}\,\text{m}^{-2}$ |
| `EFLUX` | Total latent energy flux | `tavg1_2d_flx_Nx` | `EFLUX` | $\text{W}\,\text{m}^{-2}$ |

#### Temperature

| Variable Name | Description | GEOS-5 FP Product | Raw Variable | Units |
| :--- | :--- | :--- | :--- | :--- |
| `Ta`, `Ta_K` | Air temperature at 2 m | `tavg1_2d_slv_Nx` | `T2M` | $\text{K}$ |
| `Ts`, `Ts_K` | Surface skin temperature | `tavg1_2d_slv_Nx` | `TS` | $\text{K}$ |
| `Tmin`, `Tmin_K` | Minimum air temperature at 2 m | `inst3_2d_asm_Nx` | `T2MMIN` | $\text{K}$ |

#### Pressure and Humidity

| Variable Name | Description | GEOS-5 FP Product | Raw Variable | Units |
| :--- | :--- | :--- | :--- | :--- |
| `PS` | Surface pressure | `tavg1_2d_slv_Nx` | `PS` | $\text{Pa}$ |
| `Q` | Specific humidity at 2 m | `tavg1_2d_slv_Nx` | `QV2M` | $\text{kg}\,\text{kg}^{-1}$ |

#### Wind

| Variable Name | Description | GEOS-5 FP Product | Raw Variable | Units |
| :--- | :--- | :--- | :--- | :--- |
| `U2M` | Eastward wind component at 2 m | `inst3_2d_asm_Nx` | `U2M` | $\text{m}\,\text{s}^{-1}$ |
| `V2M` | Northward wind component at 2 m | `inst3_2d_asm_Nx` | `V2M` | $\text{m}\,\text{s}^{-1}$ |

#### Atmospheric Composition

| Variable Name | Description | GEOS-5 FP Product | Raw Variable | Units |
| :--- | :--- | :--- | :--- | :--- |
| `vapor_kgsqm` | Total precipitable water vapor | `inst3_2d_asm_Nx` | `TQV` | $\text{kg}\,\text{m}^{-2}$ |
| `vapor_gccm` | Total precipitable water vapor | `inst3_2d_asm_Nx` | `TQV` | $\text{g}\,\text{cm}^{-2}$ (`TQV / 10`) |
| `ozone_dobson` | Total column ozone | `inst3_2d_asm_Nx` | `TO3` | Dobson units ($\text{DU}$) |
| `ozone_cm` | Total column ozone | `inst3_2d_asm_Nx` | `TO3` | $\text{cm-atm}$ (`TO3 / 1000`) |
| `CO2SC`, `Ca` | Atmospheric $\text{CO}_2$ surface concentration | `tavg3_2d_chm_Nx` | `CO2SC` | $\text{ppmv}$ |
| `AOT` | Aerosol optical thickness (total extinction) | `tavg3_2d_aer_Nx` | `TOTEXTTAU` | dimensionless |
| `COT` | Cloud optical thickness | `tavg1_2d_rad_Nx` | `TAUTOT` | dimensionless |

#### Radiation & Photosynthetically Active Radiation (PAR)

| Variable Name | Description | GEOS-5 FP Product | Raw Variable | Units |
| :--- | :--- | :--- | :--- | :--- |
| `SWin` | Net surface downward shortwave flux | `tavg1_2d_rad_Nx` | `SWGNT` | $\text{W}\,\text{m}^{-2}$ |
| `SWTDN` | Top-of-atmosphere incident shortwave flux | `tavg1_2d_rad_Nx` | `SWTDN` | $\text{W}\,\text{m}^{-2}$ |
| `PARDR` | Surface downward direct beam PAR flux | `tavg1_2d_lnd_Nx` | `PARDR` | $\text{W}\,\text{m}^{-2}$ |
| `PARDF` | Surface downward diffuse PAR flux | `tavg1_2d_lnd_Nx` | `PARDF` | $\text{W}\,\text{m}^{-2}$ |

#### Albedo Components

| Variable Name | Description | GEOS-5 FP Product | Raw Variable | Units |
| :--- | :--- | :--- | :--- | :--- |
| `ALBEDO` | Total surface albedo | `tavg1_2d_rad_Nx` | `ALBEDO` | fraction ($0\text{--}1$) |
| `ALBVISDR` | Direct beam visible surface albedo | `tavg1_2d_rad_Nx` | `ALBVISDR` | fraction ($0\text{--}1$) |
| `ALBVISDF` | Diffuse beam visible surface albedo | `tavg1_2d_rad_Nx` | `ALBVISDF` | fraction ($0\text{--}1$) |
| `ALBNIRDR` | Direct beam near-infrared surface albedo | `tavg1_2d_rad_Nx` | `ALBNIRDR` | fraction ($0\text{--}1$) |
| `ALBNIRDF` | Diffuse beam near-infrared surface albedo | `tavg1_2d_rad_Nx` | `ALBNIRDF` | fraction ($0\text{--}1$) |

### Computed / Derived Variables

| Variable Name | Description | Formula / Dependencies | Units |
| :--- | :--- | :--- | :--- |
| `Ta_C` | Air temperature at 2 m in Celsius | $T_a - 273.15$ (`Ta_K`) | $^{\circ}\text{C}$ |
| `Tmin_C` | Minimum air temperature at 2 m in Celsius | $T_{\min} - 273.15$ (`Tmin_K`) | $^{\circ}\text{C}$ |
| `RH` | Relative humidity (Magnus-Tetens formulation) | Derived from `Q`, `PS`, `Ta` | fraction ($0\text{--}1$) |
| `SVP_Pa` | Saturated vapor pressure | $610.8 \exp\!\left(\frac{17.27\, T_{a,C}}{T_{a,C} + 237.3}\right)$ (`Ta`) | $\text{Pa}$ |
| `Ea_Pa` | Actual vapor pressure | $\text{RH} \times \text{SVP}_{\text{Pa}}$ (`Q`, `PS`, `Ta`) | $\text{Pa}$ |
| `VPD_kPa` | Vapor pressure deficit | $\max(\text{SVP}_{\text{Pa}} - E_{a,\text{Pa}},\, 0) / 1000$ (`Q`, `PS`, `Ta`) | $\text{kPa}$ |
| `Td_K` | Dew point temperature | $T_a - (100 - 100 \cdot \text{RH}) / 5$ (`Q`, `PS`, `Ta`) | $\text{K}$ |
| `wind_speed_mps` | Horizontal wind speed magnitude at 2 m | $\sqrt{U_{2\text{M}}^2 + V_{2\text{M}}^2}$ (`U2M`, `V2M`) | $\text{m}\,\text{s}^{-1}$ |
| `PAR_proportion` | Visible / PAR direct albedo fraction | $\text{ALBVISDR} / \text{ALBEDO}$ | fraction ($0\text{--}1$) |
| `NIR_proportion` | Near-infrared direct albedo fraction | $\text{ALBNIRDR} / \text{ALBEDO}$ | fraction ($0\text{--}1$) |
| `PAR_albedo` | Combined direct + diffuse visible/PAR albedo | $\text{ALBVISDR} + \text{ALBVISDF}$ | fraction ($0\text{--}1$) |
| `NIR_albedo` | Combined direct + diffuse NIR albedo | $\text{ALBNIRDR} + \text{ALBNIRDF}$ | fraction ($0\text{--}1$) |
| `albedo_visible` | Visible surface albedo | `ALBVISDR` | fraction ($0\text{--}1$) |
| `albedo_NIR` | Near-infrared surface albedo | `ALBNIRDR` | fraction ($0\text{--}1$) |

*(Note: `NDVI` is also available as a raster method `conn.NDVI(time_UTC, geometry)` computed from `LAI` via $\text{clip}(1.05 - e^{-0.5 \cdot \text{LAI}}, 0, 1)$.)*

### Variable Aliases

| Alias | Canonical Variable | Description |
| :--- | :--- | :--- |
| `SM` | `SFMC` | Top layer soil moisture |
| `Ta` | `Ta_K` | Air temperature at 2 m in Kelvin |
| `Ts` | `Ts_K` | Surface temperature in Kelvin |
| `Tmin` | `Tmin_K` | Minimum temperature at 2 m in Kelvin |
| `Ca` | `CO2SC` | Atmospheric $\text{CO}_2$ concentration in ppmv |
| `vapor_kgsqm` / `vapor_gccm` | `TQV` | Precipitable water vapor (`vapor_gccm` scaled by $0.1$) |
| `ozone_dobson` / `ozone_cm` | `TO3` | Total column ozone (`ozone_cm` scaled by $0.001$) |

### Spatial & Temporal Resolution

- **Native Spatial Grid:** Global latitude/longitude grid (`EPSG:4326`) at $0.3125^{\circ}$ longitude $\times$ $0.25^{\circ}$ latitude ($\approx 27.4\text{ km}$).
- **Temporal Cadence:**
  - `tavg1_*`: Hourly time-averaged granules centered at `:30` past each hour (`00:30`, `01:30`, ..., `23:30` UTC).
  - `tavg3_*`: 3-hourly time-averaged granules centered at `01:30`, `04:30`, `07:30`, `10:30`, `13:30`, `16:30`, `19:30`, `22:30` UTC.
  - `inst3_*`: 3-hourly instantaneous granules at `00:00`, `03:00`, `06:00`, `09:00`, `12:00`, `15:00`, `18:00`, `21:00` UTC.

### Programmatic Variable Discovery

```python
from GEOS5FP.constants import GEOS5FP_VARIABLES, COMPUTED_VARIABLES, VARIABLE_TRANSFORMATIONS

print("Direct variables:", sorted(GEOS5FP_VARIABLES.keys()))
print("Computed variables:", sorted(COMPUTED_VARIABLES))
print("Unit transformations:", sorted(VARIABLE_TRANSFORMATIONS.keys()))
```

## Known Issues & Operational Notes

- **Gridded time-series queries:** `time_range=(start, end)` queries via `.query()` are supported for `Point`, `MultiPoint`, and `GeoDataFrame` targets. Gridded `RasterGrid` / `RasterGeometry` queries require a single `time_UTC` timestamp.
- **OPeNDAP server availability:** Point, time-series, and batch table queries rely on NASA NCCS's OPeNDAP service (`opendap.nccs.nasa.gov`), which can occasionally experience transient timeouts or `504 Gateway Time-out` errors. Both `query_geos5fp_point` and `query_geos5fp_point_multi` include automatic exponential-backoff retry logic (`retries=3`) and scoped socket timeouts; you can also initialize `GEOS5FPConnection(allow_fallback=True)` to fall back to HTTP `.nc4` granule downloads for single-point queries.
- **Granule `.lock` files:** Concurrent downloads create `<granule>.nc4.lock` files alongside cached granules in `download_directory` to prevent multi-process write collisions.

## Release History & Technical Evolution

### Architectural Eras

1. **Monolithic Prototype & Initial PyPI Release (`v1.0.1` – `v1.0.7`, Jun 2024 – Mar 2025):**
   Launched as `geos5fp` with HTTP directory scraping (`BeautifulSoup`), `wget`-based granule downloads, linear temporal interpolation between bracketing forecast/assimilation granules, and empirical thermal/moisture downscaling (`linear_downscale`, `bias_correct`, `downscale_air_temperature`, `downscale_soil_moisture`, `downscale_vapor_pressure_deficit`, `downscale_relative_humidity`). Decomposed the single-file prototype into `GEOS5FPConnection`, `GEOS5FPGranule`, `HTTP_listing`, `constants`, `exceptions`, and `Timer`, and added GitHub Actions CI.
2. **Local Cache Simplification, Uppercase Rebranding & Pure-Python Downloads (`v1.1.1` – `v1.2.4`, Mar 2025 – Oct 2025):**
   Renamed the package to uppercase `GEOS5FP`, eliminated intermediate GeoTIFF product directories and `working_directory` in favor of a single `download_directory` (`~/data/GEOS5FP`), short-circuited remote HTTP checks when valid granules already existed locally, replaced external `wget` calls with pure-Python streaming downloads, adopted `importlib.metadata` versioning, and added fast nearest-neighbor resampling for `Point` granule reads.
3. **Crash-Resilient NetCDF Validation & SSL Hardening (`v1.3.0` – `v1.3.2`, Oct 2025 – Nov 2025):**
   Built `validate_GEOS5FP_NetCDF_file.py` with subprocess isolation (`_validate_in_subprocess`) and JSON IPC to prevent low-level C++ `SIGABRT`/segfault crashes in GDAL/HDF5 from terminating parent pipelines, and added `GEOS5FPSSLError` with multi-stage TLS fallback handling for NASA NCCS connections.
4. **OPeNDAP Point Queries, Vectorized Tables & Computed Variables (`v2.0.0` – `v2.11.0`, Nov 2025 – Dec 2025):**
   Introduced `GEOS5FP_point.py` (`query_geos5fp_point` and `query_geos5fp_point_multi`) using `xarray` and `netCDF4` against NASA's OPeNDAP server, centralized variable definitions in `GEOS5FP/variables.csv`, implemented vectorized spatio-temporal batching (`targets_df`, coordinate grouping, `cluster_times`), added computed meteorological variables (`RH`, `Ta_C`, `Ea_Pa`, `SVP_Pa`, `VPD_kPa`, `Td_K`, `wind_speed_mps`, albedo proportions), added `pandas.Series` / `DatetimeIndex` / array `time_UTC` support, disabled remote CF decoding to prevent OPeNDAP hangs, and implemented 7-day temporal chunking for long time-series queries.
5. **Query Modularization, Pooch Staged Downloads, GDAL Independence & Query Parity (`v2.12.0` – `v2.25.0`, Jan 2026 – Nov 2026):**
   Extracted `query()` into `GEOS5FP/query.py`; standardized single-variable non-raster queries to return `numpy.ndarray`; fixed `tavg3` expected-hour intervals (`[1.5, 4.5, ..., 22.5]`); migrated downloads to `pooch.HTTPDownloader` with out-of-tree staging (`pooch.os_cache("GEOS5FP")`), atomic cross-device publication, and inter-process file locks (`_granule_file_lock`) to prevent cloud-sync write timeouts; added pure-`netCDF4` reading and validation fallbacks when GDAL's `gdal_HDF5.so` plugin is absent; and unified single-element sequences, multi-variable raster `dict` returns, all 14 computed variables, unit scaling, and linear temporal interpolation across all `.query()` modes.

### Complete Release Log

| Version | Date (UTC) | Summary |
| :--- | :--- | :--- |
| **`v2.25.0`** | 2026-11-23 | Unified single-element sequence & `variables=` alias handling on gridded geometries (`#60`); added multi-variable raster `dict[str, Raster]` returns (`#65`) and standalone `query()` raster support (`#66`); implemented all 14 `COMPUTED_VARIABLES` across batch, point, and time-series modes (`#61`, `#62`); applied `vapor_gccm` and `ozone_cm` unit transformations (`#63`) and linear temporal interpolation in batch queries (`#64`); scoped socket timeouts with thread locking and added retry backoff to `query_geos5fp_point_multi` (`#67`); added TTL expiration and `refresh=True` invalidation to directory listing caches (`#68`); deprecated `save_products` and hardened `HEAD` checks (`#69`); relocated root scripts to `examples/`, converted root point tests to mocked unit tests (`#70`), and moved script-only packages to optional dependencies (`#71`). |
| **`v2.24.0`** | 2026-11-17 | Added direct `netCDF4` reading (`_read_via_netcdf4`) and validation (`_validate_with_netcdf4`) fallbacks when `gdal_HDF5.so` is unavailable; fixed subprocess `ImportError` JSON serialization; added inter-process file locking (`_granule_file_lock`) and `albedo_visible`/`albedo_NIR` computed variables; set `requires-python = ">=3.11"`. |
| **`v2.23.0`** | 2026-11-10 | Enabled `pooch` download progress bar display and added `examples/download_progress.py`. |
| **`v2.22.0`** | 2026-11-10 | Replaced custom streaming download transport with `pooch.HTTPDownloader` using out-of-tree temporary staging and atomic cross-device publication (`_publish_staged_file`). |
| **`v2.21.0`** | 2026-10-12 | Fixed default `expected_hours` (`[1.5, 4.5, ..., 22.5]`) for 3-hourly `tavg3_*` products. |
| **`v2.20.0`** | 2026-07-23 | Restored single-time raster snapshot routing in `GEOS5FPConnection.query()` and fixed `MultiPoint` OPeNDAP queries. |
| **`v2.19.0`** | 2026-01-20 | Added `PAR_albedo` and `NIR_albedo` methods and constants. |
| **`v2.18.0`** | 2026-01-16 | Added `Tmin_C` computed variable. |
| **`v2.17.0`** | 2026-01-16 | Reduced batch time-clustering window to 1 day and fixed multi-dataset batch count estimation. |
| **`v2.16.0`** | 2026-01-15 | Disabled `tqdm` notebook progress bars to prevent UI freezing. |
| **`v2.15.0`** | 2026-01-14 | Extended `rasters` point wrapper unwrapping to single-time point queries. |
| **`v2.14.0`** | 2026-01-02 | Added unwrapping for `rasters` point geometry wrapper objects in batch queries. |
| **`v2.13.0`** | 2026-01-02 | Standardized single-variable non-raster queries to return 1D `numpy.ndarray`. |
| **`v2.12.0`** | 2026-01-02 | Extracted `query()` into standalone `GEOS5FP/query.py` module. |
| **`v2.11.0`** | 2025-12-30 | Implemented 7-day temporal chunking for long single-point OPeNDAP time-series queries. |
| **`v2.10.0`** | 2025-12-23 | Unified variable retrieval methods under `.query()` and added single-point geometry broadcasting. |
| **`v2.9.1`** | 2025-12-22 | Added 120s socket timeouts and retry backoff to OPeNDAP point queries. |
| **`v2.9.0`** | 2025-12-19 | Added detailed per-step progress and ETA logging for granule searches and time-series queries. |
| **`v2.8.0`** | 2025-12-18 | Added `pd.DatetimeIndex` support and suppressed timezone/NumPy warnings in tests. |
| **`v2.7.0`** | 2025-12-18 | Added support for passing `pandas.Series`, lists, tuples, and NumPy arrays as `time_UTC`. |
| **`v2.6.0`** | 2025-12-16 | Renamed visible albedo fraction to `PAR_proportion` alongside `NIR_proportion`. |
| **`v2.5.0`** | 2025-12-16 | Added visible/NIR albedo scale factors, examples, and expanded `README.md` documentation. |
| **`v2.4.0`** | 2025-12-10 | Added `Ca` alias, `wind_speed_mps`, and `decode_cf=False` OPeNDAP optimization to eliminate remote hangs. |
| **`v2.3.0`** | 2025-11-25 | Added elapsed time and ETA progress logging/bars for batch and point queries. |
| **`v2.1.0`** | 2025-11-25 | Added `targets_df` table enrichment, `cluster_times`, `calculate_RH`, computed variables (`RH`, `Ta_C`, etc.), SSL helper modules, and `lanczos` default resampling. |
| **`v2.0.1`** | 2025-11-25 | Suppressed `NumExpr` thread-count log messages on package import. |
| **`v2.0.0`** | 2025-11-24 | Major release: OPeNDAP point/time-series queries (`GEOS5FP_point.py`), CSV-driven variable registry (`variables.csv`), vectorized table queries, and unit test suite. |
| **`v1.3.2`** | 2025-11-05 | Added multi-stage SSL fallback including legacy `DEFAULT@SECLEVEL=1` cipher context. |
| **`v1.3.1`** | 2025-10-28 | Added `GEOS5FPSSLError` and automatic SSL verification fallback for NCCS connections. |
| **`v1.3.0`** | 2025-10-27 | Added comprehensive NetCDF validation (`validate_GEOS5FP_NetCDF_file`) with subprocess isolation against C++/GDAL crashes. |
| **`v1.2.4`** | 2025-10-24 | Added nearest-neighbor resampling when reading granules for `Point` geometries. |
| **`v1.2.3`** | 2025-08-04 | Added `ChunkedEncodingError` and `ConnectionError` retry handling; switched to `importlib.metadata` versioning. |
| **`v1.2.2`** | 2025-07-22 | Reduced verbose directory initialization logging. |
| **`v1.2.1`** | 2025-07-22 | Cleaned up unused constant imports in `GEOS5FP_granule.py`. |
| **`v1.2.0`** | 2025-07-22 | Removed `working_directory`; defaulted `download_directory` to `~/data/GEOS5FP`; replaced `wget` with `requests` + `tqdm`; improved date/time string parsing. |
| **`v1.1.2`** | 2025-07-15 | Reduced redundant remote HTTP `HEAD` requests when granules are already cached locally. |
| **`v1.1.1`** | 2025-03-28 | Renamed package to uppercase `GEOS5FP`; skipped remote checks when local files exist; removed intermediate product files. |
| **`v1.0.7`** | 2025-03-10 | Added `brew` and `mamba` installation instructions for `wget`. |
| **`v1.0.6`** | 2025-03-10 | Updated README installation instructions for `wget`. |
| **`v1.0.5`** | 2025-02-28 | Standardized module-level loggers across connection and granule classes. |
| **`v1.0.4`** | 2025-02-26 | Exported `constants` and `exceptions` at package top level. |
| **`v1.0.3`** | 2025-02-26 | Fixed relative imports, added dependency import tests, and configured GitHub Actions CI. |
| **`v1.0.2`** | 2025-02-26 | Modularized monolithic script into `GEOS5FPConnection`, `GEOS5FPGranule`, `HTTP_listing`, `constants`, and `exceptions`. |
| **`v1.0.1`** | 2024-07-20 | Initial PyPI release (`geos5fp`); HTTP directory listing, `wget` download, temporal interpolation, and spatial downscaling. |

## Data Source & Data Product Citation

This package accesses GEOS-5 FP (Forward Processing) data produced by the Global Modeling and Assimilation Office (GMAO) at NASA Goddard Space Flight Center.

### Data Access

GEOS-5 FP data is accessed through:
- **OPeNDAP Server:** `https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/`
- **HTTP Server:** `https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das`

Data is provided by NASA's Center for Climate Simulation (NCCS).

### Data Product Citation

When using GEOS-5 FP data in publications, please also cite:

**Data Product:**
```text
Global Modeling and Assimilation Office (GMAO) (2015), GEOS-5 FP: GEOS Forward 
Processing for Instrument Support, Greenbelt, MD, USA, Goddard Earth Sciences 
Data and Information Services Center (GES DISC). 
Accessed: [Date]
```

**Acknowledgment:**
```text
GEOS-5 FP data used in this study were provided by the Global Modeling and 
Assimilation Office (GMAO) at NASA Goddard Space Flight Center through the 
NASA Center for Climate Simulation (NCCS).
```

For more information about GEOS-5 FP, visit: https://gmao.gsfc.nasa.gov/GEOS/