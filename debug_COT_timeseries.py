"""
Debug Script: Time-series query of Cloud Optical Thickness (COT) for Los Angeles.

This script demonstrates how to retrieve a month-long time-series of Cloud Optical 
Thickness using the .COT() method.
"""

import warnings
from datetime import datetime, timedelta
import pandas as pd

# Suppress xarray SerializationWarning about ambiguous reference dates
warnings.filterwarnings('ignore', message='.*Ambiguous reference date string.*')

try:
    from GEOS5FP import GEOS5FP
    
    print("=" * 70)
    print("Time-Series Query: Cloud Optical Thickness (COT)")
    print("Location: Los Angeles (34.05°N, 118.25°W)")
    print("Duration: 1 month")
    print("=" * 70)
    
    # Define coordinates for Los Angeles
    lat = 34.05
    lon = -118.25
    
    # Define time range for one month
    # Using a recent date range that should have data available
    end_time = datetime(2024, 11, 15, 0, 0)
    start_time = end_time - timedelta(days=180)
    
    print(f"\nQuerying COT data from {start_time} to {end_time}...")
    print("Location: Latitude={}, Longitude={}".format(lat, lon))
    print("-" * 70)
    
    # Create GEOS5FP connection
    geos = GEOS5FP()
    
    print("\nCalling .COT() method with time_range parameter...\n")
    
    # Query COT using the .COT() method with time_range
    # Note: The .COT() method signature doesn't support time_range directly,
    # so we need to use .query() instead
    df = geos.query(
        target_variables="COT",
        lat=lat,
        lon=lon,
        time_range=(start_time, end_time),
        dropna=True
    )
    
    print(f"\n✓ Query completed!")
    print(f"✓ Retrieved {len(df)} COT records")
    
    print("\n" + "=" * 70)
    print("Complete Time-Series DataFrame:")
    print("=" * 70)
    print(df)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total records retrieved: {len(df)}")
    print(f"Valid COT values: {df['COT'].notna().sum()}")
    print(f"Missing values: {df['COT'].isna().sum()}")
    
    # Display statistics
    print("\nCOT Statistics:")
    print(df['COT'].describe())
    
    # Display first and last few records
    print("\nFirst 5 records:")
    print(df.head())
    
    print("\nLast 5 records:")
    print(df.tail())
    
    # Save to CSV
    output_file = "COT_timeseries_LA.csv"
    df.to_csv(output_file)
    print(f"\n✓ Results saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("Query completed successfully!")
    print("=" * 70)
    
except ImportError as e:
    print(f"ERROR: Could not import GEOS5FP module: {e}")
    print("Make sure the GEOS5FP package is installed and available.")
    
except Exception as e:
    print(f"\nERROR: An unexpected error occurred:")
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
