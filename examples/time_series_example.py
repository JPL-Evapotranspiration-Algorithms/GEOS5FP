"""
Example: Time-series query of air temperature in Celsius for Los Angeles.

This example demonstrates how to retrieve a week-long time-series of air 
temperature in Celsius using a single OPeNDAP query with a time range.
"""

import warnings
from datetime import datetime, timedelta
import pandas as pd

# Suppress xarray SerializationWarning about ambiguous reference dates
warnings.filterwarnings('ignore', message='.*Ambiguous reference date string.*')

try:
    # Import the direct OPeNDAP query function for efficient time-series retrieval
    from GEOS5FP.GEOS5FP_point import query_geos5fp_point
    
    print("=" * 70)
    print("Time-Series Query: Air Temperature in Celsius")
    print("Location: Los Angeles (34.05°N, 118.25°W)")
    print("Duration: 1 week")
    print("=" * 70)
    
    # Define coordinates for Los Angeles
    lat = 34.05
    lon = -118.25
    
    # Define time range for the past week
    # Using a date range that should have data available
    end_time = datetime(2024, 11, 15, 0, 0)
    start_time = end_time - timedelta(days=7)
    
    print(f"\nQuerying data from {start_time} to {end_time}...")
    print("Making a single OPeNDAP call with time-slice...")
    print("-" * 70)
    
    # Make a single OPeNDAP query for the entire time range
    # This is MUCH faster than querying time-step by time-step
    result = query_geos5fp_point(
        dataset="tavg1_2d_slv_Nx",  # 1-hourly time-averaged, 2D, single-level
        variable="t2m",              # 2-meter air temperature
        lat=lat,
        lon=lon,
        time_range=(start_time, end_time),
        dropna=True
    )
    
    # The result DataFrame is already time-indexed with all values
    df = result.df.copy()
    
    # Convert from Kelvin to Celsius
    df['temperature_C'] = df['t2m'] - 273.15
    
    # Add coordinate information
    df['lat'] = result.lat_used
    df['lon'] = result.lon_used
    
    # Remove the Kelvin column
    df = df.drop(columns=['t2m'])
    
    print(f"✓ Retrieved {len(df)} temperature records")
    
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
