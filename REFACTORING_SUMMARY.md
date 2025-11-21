# GEOS5FP Refactoring Summary

## Overview
Refactored the GEOS5FP package to centralize NAME/PRODUCT/VARIABLE constants in a CSV file, making the code more maintainable and reducing duplication.

## Changes Made

### 1. Variables CSV File (`GEOS5FP/variables.csv`) - **NEW**
Created a CSV file containing all variable mappings:
- **Format**: CSV with columns: `variable_name`, `description`, `product`, `variable`
- **Contains**: All 31 variable mappings including aliases
- **Aliases supported**: SM/SFMC, Ts/Ts_K, Ta/Ta_K, Tmin/Tmin_K, vapor_kgsqm/vapor_gccm, ozone_dobson/ozone_cm
- **Benefit**: Easy to edit in spreadsheet applications or text editors

### 2. Constants File (`GEOS5FP/constants.py`)
Updated to load variable mappings from CSV file:
- Added `_load_variables()` function that reads `variables.csv`
- `GEOS5FP_VARIABLES` dictionary is now populated from CSV at import time
- **Structure maintained**: `{variable_name: (description, product, variable)}`

### 3. Package Configuration (`pyproject.toml`)
Updated to include CSV file in package distribution:
- Added `*.csv` to `[tool.setuptools.package-data]`
- Ensures `variables.csv` is included when package is installed

### 4. Connection File (`GEOS5FP/GEOS5FP_connection.py`)
#### Added Helper Method
- `_get_variable_info(variable_name)`: Looks up variable metadata from `GEOS5FP_VARIABLES`
  - Returns: tuple of (description, product, variable)
  - Raises: `KeyError` if variable not found

#### Refactored Methods (25 methods updated)
All variable retrieval methods now use `_get_variable_info()` instead of hardcoded constants:
- SFMC, LAI, LHLAND, EFLUX, PARDR, PARDF, AOT, COT
- Ts_K, Ta_K, Tmin_K, PS, Q
- vapor_kgsqm, ozone_dobson, U2M, V2M, CO2SC
- SWin, SWTDN, ALBVISDR, ALBVISDF, ALBNIRDF, ALBNIRDR, ALBEDO

### 5. Tests
Created two comprehensive test suites:

#### `tests/test_variable_constants.py` (6 tests)
- Constants dictionary exists and is properly structured
- All expected variables are present
- Variable info tuples have correct format
- `_get_variable_info()` method works correctly
- Invalid variables raise appropriate errors
- Aliases point to the same underlying data

#### `tests/test_csv_loading.py` (5 tests) - **NEW**
- CSV file exists in package
- CSV has correct format and required columns
- Loaded constants match CSV content
- CSV contains all expected variables
- `_load_variables()` function works correctly

## Benefits

1. **Human-Editable Format**: CSV can be edited in Excel, Google Sheets, or any text editor
2. **Single Source of Truth**: All variable metadata is now defined in one CSV file
3. **Easier Maintenance**: Adding new variables requires only adding a CSV row
4. **No Code Changes**: Adding variables doesn't require modifying Python code
5. **Version Control Friendly**: CSV diffs are easy to review
6. **Reduced Code**: Eliminated ~75 lines of repetitive constant definitions
7. **Better Testability**: Variable mappings and CSV loading can be tested independently
8. **Type Safety**: Helper method provides consistent interface for variable lookup
9. **Documentation**: CSV file serves as comprehensive reference for all available variables

## Backward Compatibility
✅ All existing functionality preserved
✅ All method signatures unchanged
✅ All aliases still work (SM, Ts_K, Ta_K, etc.)
✅ All 27 tests pass (16 original + 6 constants + 5 CSV loading)

## Variable Mappings Summary

| Variable Name(s) | Description | Product | Variable |
|-----------------|-------------|---------|----------|
| SM, SFMC | Top layer soil moisture | tavg1_2d_lnd_Nx | SFMC |
| LAI | Leaf area index | tavg1_2d_lnd_Nx | LAI |
| LHLAND | Latent heat flux land | tavg1_2d_lnd_Nx | LHLAND |
| EFLUX | Total latent energy flux | tavg1_2d_flx_Nx | EFLUX |
| PARDR | PAR direct beam | tavg1_2d_lnd_Nx | PARDR |
| PARDF | PAR diffuse | tavg1_2d_lnd_Nx | PARDF |
| AOT | Aerosol optical thickness | tavg3_2d_aer_Nx | TOTEXTTAU |
| COT | Cloud optical thickness | tavg1_2d_rad_Nx | TAUTOT |
| Ts, Ts_K | Surface temperature | tavg1_2d_slv_Nx | TS |
| Ta, Ta_K | Air temperature | tavg1_2d_slv_Nx | T2M |
| Tmin, Tmin_K | Minimum temperature | inst3_2d_asm_Nx | T2MMIN |
| PS | Surface pressure | tavg1_2d_slv_Nx | PS |
| Q | Specific humidity | tavg1_2d_slv_Nx | QV2M |
| vapor_kgsqm, vapor_gccm | Water vapor | inst3_2d_asm_Nx | TQV |
| ozone_dobson, ozone_cm | Ozone | inst3_2d_asm_Nx | TO3 |
| U2M | Eastward wind | inst3_2d_asm_Nx | U2M |
| V2M | Northward wind | inst3_2d_asm_Nx | V2M |
| CO2SC | CO2 surface concentration | tavg3_2d_chm_Nx | CO2SC |
| SWin | Incoming shortwave radiation | tavg1_2d_rad_Nx | SWGNT |
| SWTDN | Top of atmosphere SW | tavg1_2d_rad_Nx | SWTDN |
| ALBVISDR | Direct visible albedo | tavg1_2d_rad_Nx | ALBVISDR |
| ALBVISDF | Diffuse visible albedo | tavg1_2d_rad_Nx | ALBVISDF |
| ALBNIRDF | Diffuse NIR albedo | tavg1_2d_rad_Nx | ALBNIRDF |
| ALBNIRDR | Direct NIR albedo | tavg1_2d_rad_Nx | ALBNIRDR |
| ALBEDO | Surface albedo | tavg1_2d_rad_Nx | ALBEDO |

## Example Usage

```python
from GEOS5FP import GEOS5FPConnection
from GEOS5FP.constants import GEOS5FP_VARIABLES

# Create connection
conn = GEOS5FPConnection()

# Look up variable info directly from constants
description, product, variable = GEOS5FP_VARIABLES["SFMC"]
print(f"SFMC: {description} from {product}.{variable}")

# Or use internally via the connection methods
soil_moisture = conn.SFMC("2023-01-01")  # Uses centralized constants automatically
```
