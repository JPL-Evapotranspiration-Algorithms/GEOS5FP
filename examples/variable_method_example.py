"""
Example: Using the generalized `variable` method to retrieve GEOS-5 FP data.

This example demonstrates the flexible `variable()` method which can:
1. Query any variable from any dataset
2. Retrieve time-series for point locations using efficient OPeNDAP queries
3. Work with both predefined variable names and raw GEOS-5 FP variable names
"""

import warnings
from datetime import datetime, timedelta
import pandas as pd

# Suppress xarray warnings
warnings.filterwarnings('ignore', message='.*Ambiguous reference date string.*')

try:
    from GEOS5FP import GEOS5FPConnection
    
    # Create connection
    conn = GEOS5FPConnection()
    
    print("=" * 80)
    print("GEOS-5 FP Generalized Variable Retrieval Examples")
    print("=" * 80)
    
    # Define time range
    end_time = datetime(2024, 11, 15, 0, 0)
    start_time = end_time - timedelta(days=3)  # 3 days of data
    
    # Los Angeles coordinates
    lat = 34.05
    lon = -118.25
    
    # =========================================================================
    # Example 1: Time-series query with predefined variable name
    # =========================================================================
    print("\n" + "=" * 80)
    print("Example 1: Air Temperature Time-Series (Predefined Variable)")
    print("=" * 80)
    print(f"Location: Los Angeles ({lat}°N, {lon}°W)")
    print(f"Time range: {start_time} to {end_time}")
    print(f"Variable: Ta_K (air temperature in Kelvin)")
    
    df_temp = conn.query(
        "Ta_K",
        time_range=(start_time, end_time),
        lat=lat,
        lon=lon
    )
    
    # Convert to Celsius
    df_temp['Ta_C'] = df_temp['Ta_K'] - 273.15
    
    print(f"\n✓ Retrieved {len(df_temp)} records")
    print(f"\nFirst 5 records:")
    print(df_temp[['Ta_K', 'Ta_C']].head())
    print(f"\nTemperature statistics (°C):")
    print(f"  Mean: {df_temp['Ta_C'].mean():.2f}°C")
    print(f"  Min:  {df_temp['Ta_C'].min():.2f}°C")
    print(f"  Max:  {df_temp['Ta_C'].max():.2f}°C")
    
    # =========================================================================
    # Example 2: Time-series query with raw variable name
    # =========================================================================
    print("\n" + "=" * 80)
    print("Example 2: Specific Humidity Time-Series (Raw Variable Name)")
    print("=" * 80)
    print(f"Variable: QV2M (2-meter specific humidity)")
    print(f"Dataset: tavg1_2d_slv_Nx")
    
    df_humidity = conn.query(
        "QV2M",  # Raw GEOS-5 FP variable name
        time_range=(start_time, end_time),
        dataset="tavg1_2d_slv_Nx",
        lat=lat,
        lon=lon
    )
    
    print(f"\n✓ Retrieved {len(df_humidity)} records")
    print(f"\nFirst 5 records:")
    print(df_humidity[['QV2M']].head())
    print(f"\nHumidity statistics (kg/kg):")
    print(f"  Mean: {df_humidity['QV2M'].mean():.6f}")
    print(f"  Min:  {df_humidity['QV2M'].min():.6f}")
    print(f"  Max:  {df_humidity['QV2M'].max():.6f}")
    
    # =========================================================================
    # Example 3: Query multiple variables and combine
    # =========================================================================
    print("\n" + "=" * 80)
    print("Example 3: Multiple Variables Combined")
    print("=" * 80)
    print("Retrieving: Temperature, Wind components, and Solar radiation")
    
    # Temperature (already have this)
    
    # Wind components
    df_u = conn.query(
        "U2M",
        time_range=(start_time, end_time),
        dataset="tavg1_2d_slv_Nx",
        lat=lat,
        lon=lon
    )
    
    df_v = conn.query(
        "V2M",
        time_range=(start_time, end_time),
        dataset="tavg1_2d_slv_Nx",
        lat=lat,
        lon=lon
    )
    
    # Solar radiation
    df_solar = conn.query(
        "SWGDN",  # Surface incoming shortwave flux
        time_range=(start_time, end_time),
        dataset="tavg1_2d_rad_Nx",
        lat=lat,
        lon=lon
    )
    
    # Combine into single DataFrame
    df_combined = pd.DataFrame({
        'temperature_C': df_temp['Ta_K'] - 273.15,
        'wind_u': df_u['U2M'],
        'wind_v': df_v['V2M'],
        'solar_radiation': df_solar['SWGDN']
    })
    
    # Calculate wind speed
    df_combined['wind_speed'] = (df_combined['wind_u']**2 + df_combined['wind_v']**2)**0.5
    
    print(f"\n✓ Retrieved data for all variables")
    print(f"\nCombined DataFrame (first 5 records):")
    print(df_combined[['temperature_C', 'wind_speed', 'solar_radiation']].head())
    
    print(f"\nWind Speed Statistics (m/s):")
    print(f"  Mean: {df_combined['wind_speed'].mean():.2f}")
    print(f"  Max:  {df_combined['wind_speed'].max():.2f}")
    
    print(f"\nSolar Radiation Statistics (W/m²):")
    print(f"  Mean: {df_combined['solar_radiation'].mean():.2f}")
    print(f"  Max:  {df_combined['solar_radiation'].max():.2f}")
    
    # =========================================================================
    # Example 4: Different location
    # =========================================================================
    print("\n" + "=" * 80)
    print("Example 4: Different Location (New York)")
    print("=" * 80)
    
    ny_lat = 40.73
    ny_lon = -73.94
    
    print(f"Location: New York ({ny_lat}°N, {ny_lon}°W)")
    print(f"Variable: T2M (2-meter air temperature)")
    
    df_ny = conn.query(
        "T2M",
        time_range=(start_time, end_time),
        dataset="tavg1_2d_slv_Nx",
        lat=ny_lat,
        lon=ny_lon
    )
    
    df_ny['T2M_C'] = df_ny['T2M'] - 273.15
    
    print(f"\n✓ Retrieved {len(df_ny)} records")
    print(f"\nTemperature statistics (°C):")
    print(f"  Mean: {df_ny['T2M_C'].mean():.2f}°C")
    print(f"  Min:  {df_ny['T2M_C'].min():.2f}°C")
    print(f"  Max:  {df_ny['T2M_C'].max():.2f}°C")
    
    # Compare LA vs NY
    print(f"\nTemperature Comparison:")
    print(f"  Los Angeles mean: {df_temp['Ta_C'].mean():.2f}°C")
    print(f"  New York mean:    {df_ny['T2M_C'].mean():.2f}°C")
    print(f"  Difference:       {df_temp['Ta_C'].mean() - df_ny['T2M_C'].mean():.2f}°C")
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
    print("\nKey Benefits of the `variable()` method:")
    print("  ✓ Single interface for any variable from any dataset")
    print("  ✓ Efficient OPeNDAP queries for time-series data")
    print("  ✓ Works with both predefined and raw variable names")
    print("  ✓ Flexible: supports raster and point queries")
    print("  ✓ Fast: retrieves entire time ranges in single calls")
    
except ImportError as e:
    print(f"ImportError: {e}")
    print("\nTo use time-series query functionality, install required packages:")
    print("  conda install -c conda-forge xarray netcdf4")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    print("\nNote: This example requires internet connection to query GEOS-5 FP OPeNDAP server")
