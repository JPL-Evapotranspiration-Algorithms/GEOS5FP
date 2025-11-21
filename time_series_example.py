"""
Example: Time-series query of air temperature in Celsius for Los Angeles.

This example demonstrates how to retrieve a week-long time-series of air 
temperature in Celsius (Ta_C) for a specific location in Los Angeles.
"""

from shapely.geometry import Point
from datetime import datetime, timedelta
import pandas as pd

try:
    from GEOS5FP import GEOS5FPConnection
    
    # Create connection
    conn = GEOS5FPConnection()
    
    print("=" * 70)
    print("Time-Series Query: Air Temperature in Celsius")
    print("Location: Los Angeles (34.05°N, 118.25°W)")
    print("Duration: 1 week")
    print("=" * 70)
    
    # Create a point for Los Angeles (lon, lat format for shapely)
    la_point = Point(-118.25, 34.05)
    
    # Define time range for the past week
    # Using a date range that should have data available
    end_time = datetime(2024, 11, 15, 0, 0)
    start_time = end_time - timedelta(days=7)
    
    # Collect temperature data for each day
    temperature_data = []
    
    print(f"\nQuerying data from {start_time} to {end_time}...")
    print("-" * 70)
    
    # Query at 3-hour intervals (8 times per day) for a week
    current_time = start_time
    while current_time <= end_time:
        try:
            # Use Ta_K method to get temperature in Kelvin, then convert to Celsius
            result = conn.Ta_K(time_UTC=current_time, geometry=la_point)
            
            # Extract the temperature value and convert to Celsius
            if not result.empty:
                temp_k = result['Ta_K'].iloc[0]
                temp_c = temp_k - 273.15
                temperature_data.append({
                    'time': current_time,
                    'temperature_C': temp_c,
                    'lat': result['lat'].iloc[0],
                    'lon': result['lon'].iloc[0]
                })
                print(f"✓ {current_time}: {temp_c:.2f}°C")
            else:
                print(f"✗ {current_time}: No data available")
                
        except Exception as e:
            print(f"✗ {current_time}: Error - {e}")
        
        # Move to next 3-hour interval
        current_time += timedelta(hours=3)
    
    # Create DataFrame from collected data
    if temperature_data:
        df = pd.DataFrame(temperature_data)
        df.set_index('time', inplace=True)
        
        print("\n" + "=" * 70)
        print("Complete Time-Series DataFrame:")
        print("=" * 70)
        print(df)
        
        print("\n" + "=" * 70)
        print("Summary Statistics:")
        print("=" * 70)
        print(f"Mean Temperature: {df['temperature_C'].mean():.2f}°C")
        print(f"Min Temperature:  {df['temperature_C'].min():.2f}°C")
        print(f"Max Temperature:  {df['temperature_C'].max():.2f}°C")
        print(f"Std Deviation:    {df['temperature_C'].std():.2f}°C")
        print(f"Total Records:    {len(df)}")
    else:
        print("\n✗ No temperature data was successfully retrieved.")
    
    print("\n" + "=" * 70)
    print("Query completed!")
    print("=" * 70)
    
except ImportError as e:
    print(f"ImportError: {e}")
    print("\nTo use time-series query functionality, install required packages:")
    print("  conda install -c conda-forge xarray netcdf4")
except Exception as e:
    print(f"Error: {e}")
    print("\nNote: This example requires internet connection to query GEOS-5 FP OPeNDAP server")
