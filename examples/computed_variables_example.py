"""
Example demonstrating querying computed variables from GEOS-5 FP data.

Computed variables are automatically derived from base GEOS-5 FP variables
without requiring manual calculation.
"""

import geopandas as gpd
import pandas as pd
from datetime import datetime
from shapely.geometry import Point
from GEOS5FP import GEOS5FP

def main():
    # Create a connection to GEOS-5 FP
    conn = GEOS5FP()
    
    # Define test location and time
    test_location = gpd.GeoDataFrame({
        'time_UTC': [datetime(2019, 10, 2, 12, 0)],
        'geometry': [Point(-118.25, 34.05)]  # Los Angeles
    }, crs='EPSG:4326')
    
    print("=" * 60)
    print("GEOS-5 FP Computed Variables Example")
    print("=" * 60)
    print(f"\nQuerying location: Los Angeles ({34.05}°N, {-118.25}°E)")
    print(f"Time: {test_location['time_UTC'].iloc[0]}")
    
    # Example 1: Query wind speed (computed from U2M and V2M components)
    print("\n" + "-" * 60)
    print("Example 1: Wind Speed")
    print("-" * 60)
    result = conn.query(['wind_speed_mps'], test_location, verbose=False)
    print(f"Wind speed: {result['wind_speed_mps'].iloc[0]:.2f} m/s")
    print("(Computed from U2M and V2M wind components)")
    
    # Example 2: Query temperature in Celsius (computed from Kelvin)
    print("\n" + "-" * 60)
    print("Example 2: Temperature in Celsius")
    print("-" * 60)
    result = conn.query(['Ta_C'], test_location, verbose=False)
    print(f"Temperature: {result['Ta_C'].iloc[0]:.2f} °C")
    print("(Computed from Ta_K)")
    
    # Example 3: Query relative humidity (computed from Q, PS, Ta)
    print("\n" + "-" * 60)
    print("Example 3: Relative Humidity")
    print("-" * 60)
    result = conn.query(['RH'], test_location, verbose=False)
    print(f"Relative humidity: {result['RH'].iloc[0]:.1f}%")
    print("(Computed from Q, PS, and Ta)")
    
    # Example 4: Query multiple computed variables at once
    print("\n" + "-" * 60)
    print("Example 4: Multiple Computed Variables")
    print("-" * 60)
    computed_vars = ['wind_speed_mps', 'Ta_C', 'RH']
    result = conn.query(computed_vars, test_location, verbose=False)
    
    print("\nResults:")
    for var in computed_vars:
        val = result[var].iloc[0]
        if var == 'wind_speed_mps':
            print(f"  Wind speed: {val:.2f} m/s")
        elif var == 'Ta_C':
            print(f"  Temperature: {val:.2f} °C")
        elif var == 'RH':
            print(f"  Relative humidity: {val:.1f}%")
    
    # Example 5: Mix computed and base variables
    print("\n" + "-" * 60)
    print("Example 5: Mix Computed and Base Variables")
    print("-" * 60)
    mixed_vars = ['COT', 'AOT', 'Ca', 'wind_speed_mps', 'Ta_C']
    result = conn.query(mixed_vars, test_location, verbose=False)
    
    print("\nResults:")
    print(f"  Cloud optical thickness (COT): {result['COT'].iloc[0]:.2f}")
    print(f"  Aerosol optical thickness (AOT): {result['AOT'].iloc[0]:.4f}")
    print(f"  Atmospheric CO2 (Ca): {result['Ca'].iloc[0]:.2f} ppmv")
    print(f"  Wind speed: {result['wind_speed_mps'].iloc[0]:.2f} m/s (computed)")
    print(f"  Temperature: {result['Ta_C'].iloc[0]:.2f} °C (computed)")
    
    # Example 6: Multiple locations
    print("\n" + "-" * 60)
    print("Example 6: Multiple Locations")
    print("-" * 60)
    multi_locations = gpd.GeoDataFrame({
        'time_UTC': [datetime(2019, 10, 2, 12, 0)] * 3,
        'geometry': [
            Point(-118.25, 34.05),  # Los Angeles
            Point(-122.42, 37.78),  # San Francisco
            Point(-87.63, 41.88)    # Chicago
        ]
    }, crs='EPSG:4326')
    
    result = conn.query(['wind_speed_mps', 'Ta_C'], multi_locations, verbose=False)
    
    locations = ['Los Angeles', 'San Francisco', 'Chicago']
    print("\nResults:")
    for i, location in enumerate(locations):
        wind = result['wind_speed_mps'].iloc[i]
        temp = result['Ta_C'].iloc[i]
        print(f"  {location}: {wind:.2f} m/s, {temp:.2f} °C")
    
    print("\n" + "=" * 60)
    print("Available Computed Variables:")
    print("=" * 60)
    print("  - wind_speed_mps: Wind speed in m/s")
    print("  - Ta_C: Air temperature in Celsius")
    print("  - RH: Relative humidity (%)")
    print("  - VPD_kPa: Vapor pressure deficit in kPa")
    print("  - Ea_Pa: Actual vapor pressure in Pascals")
    print("  - SVP_Pa: Saturated vapor pressure in Pascals")
    print("  - Td_K: Dew point temperature in Kelvin")
    print("\nThese are automatically computed from base GEOS-5 FP variables.")
    print("No manual calculation required!")
    print("=" * 60)

if __name__ == "__main__":
    main()
