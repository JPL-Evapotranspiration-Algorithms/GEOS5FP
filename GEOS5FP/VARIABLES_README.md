# GEOS-5 FP Variables Reference

This CSV file contains the mapping of all GEOS-5 FP variables used in the package.

## Format

The CSV has 4 columns:

1. **variable_name**: The name used in the code (e.g., `SFMC`, `Ta_K`, `SM`)
2. **description**: Human-readable description of the variable
3. **product**: The GEOS-5 FP product name (e.g., `tavg1_2d_lnd_Nx`)
4. **variable**: The actual variable name in the GEOS-5 FP product (e.g., `SFMC`, `T2M`)

## Aliases

Some variables have multiple names (aliases) that point to the same underlying data:
- `SM` and `SFMC` → Soil Moisture
- `Ts` and `Ts_K` → Surface Temperature
- `Ta` and `Ta_K` → Air Temperature
- `Tmin` and `Tmin_K` → Minimum Temperature
- `vapor_kgsqm` and `vapor_gccm` → Water Vapor
- `ozone_dobson` and `ozone_cm` → Ozone
- `Ca` and `CO2SC` → Atmospheric CO2 Concentration

## Computed Variables

The package automatically computes derived variables from base GEOS-5 FP data:

- **`RH`** - Relative Humidity (computed from Q, PS, Ta)
- **`Ta_C`** - Air Temperature in Celsius (computed from Ta)
- **`Ea_Pa`** - Actual Vapor Pressure in Pascals (computed from RH and SVP)
- **`SVP_Pa`** - Saturated Vapor Pressure in Pascals (computed from Ta)
- **`VPD_kPa`** - Vapor Pressure Deficit in kPa (computed from SVP and Ea)
- **`Td_K`** - Dew Point Temperature in Kelvin (computed from Ta and RH)
- **`wind_speed_mps`** - Wind Speed in m/s (computed from U2M and V2M)

When you query a computed variable, the package automatically:
1. Identifies the required base variables
2. Queries them from the appropriate GEOS-5 FP products
3. Computes the derived variable
4. Returns only the requested variable (base variables are not included in output)

## Adding New Variables

To add a new variable:

1. Open `variables.csv` in a spreadsheet application or text editor
2. Add a new row with:
   - variable_name: The name you want to use in the code
   - description: Brief description of the variable
   - product: The GEOS-5 FP product containing this variable
   - variable: The exact variable name in the product
3. Save the file
4. Add a corresponding method in `GEOS5FP_connection.py` if needed

No other code changes are required! The constants are automatically loaded from this CSV file.

## Products Reference

Common GEOS-5 FP products:
- `tavg1_2d_lnd_Nx` - Hourly time-averaged land surface diagnostics
- `tavg1_2d_slv_Nx` - Hourly time-averaged single-level diagnostics
- `tavg1_2d_flx_Nx` - Hourly time-averaged surface flux diagnostics
- `tavg1_2d_rad_Nx` - Hourly time-averaged radiation diagnostics
- `tavg3_2d_aer_Nx` - 3-hourly time-averaged aerosol diagnostics
- `tavg3_2d_chm_Nx` - 3-hourly time-averaged chemistry diagnostics
- `inst3_2d_asm_Nx` - 3-hourly instantaneous assimilated meteorology

## Maintenance

This file is the single source of truth for all GEOS-5 FP variable mappings in the package. Keep it synchronized with the actual available GEOS-5 FP products and variables.

## Testing

The file is validated by tests in `tests/test_csv_loading.py` which ensure:
- CSV file exists and has correct format
- All expected variables are present
- Loaded constants match CSV content
- CSV structure is valid
