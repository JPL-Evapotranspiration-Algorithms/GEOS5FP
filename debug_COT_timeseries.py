"""
Debug Script: Time-series query of Cloud Optical Thickness (COT) for Los Angeles.

This script demonstrates how to retrieve a week-long time-series of Cloud Optical 
Thickness using direct OPeNDAP queries for efficient time-series retrieval.

Note: There's a bug in GEOS5FP_connection.py around line 2635 where query_geos5fp_point
is called but not locally imported. This script uses the direct query_geos5fp_point 
function as a workaround.
"""

import warnings
from datetime import datetime, timedelta
import pandas as pd

# Suppress xarray SerializationWarning about ambiguous reference dates
warnings.filterwarnings('ignore', message='.*Ambiguous reference date string.*')

try:
    from GEOS5FP.GEOS5FP_point import query_geos5fp_point
    
    print("=" * 70)
    print("Time-Series Query: Cloud Optical Thickness (COT)")
    print("Location: Los Angeles (34.05°N, 118.25°W)")
    print("Duration: 1 week (single OPeNDAP query with time_range)")
    print("=" * 70)
    
    # Define coordinates for Los Angeles
    lat = 34.05
    lon = -118.25
    
    # Define time range for one week
    # Using a recent date range that should have data available
    end_time = datetime(2024, 11, 15, 0, 0)
    start_time = end_time - timedelta(days=7)
    
    print(f"\nQuerying COT data from {start_time} to {end_time}...")
    print("Location: Latitude={}, Longitude={}".format(lat, lon))
    print("Using single OPeNDAP query with time_range parameter")
    print("-" * 70)
    
    print("\nMaking single OPeNDAP call with time-slice...")
    print("Testing direct query_geos5fp_point function...\n")
    
    # Query COT using direct OPeNDAP query function
    # COT is in dataset tavg1_2d_rad_Nx, variable TAUTOT
    result = query_geos5fp_point(
        dataset="tavg1_2d_rad_Nx",
        variable="tautot",  # lowercase for OPeNDAP
        lat=lat,
        lon=lon,
        time_range=(start_time, end_time),
        dropna=True
    )
    
    # Result is a PointQueryResult with a df attribute
    print(f"✓ Query completed!")
    print(f"Result type: {type(result)}")
    print(f"✓ Retrieved {len(result.df)} COT records")
    
    # Extract the DataFrame
    df = result.df.copy()
    
    # Rename the column from 'tautot' to 'COT'
    if 'tautot' in df.columns:
        df = df.rename(columns={'tautot': 'COT'})
    
    # Add coordinate information
    df['lat'] = result.lat_used
    df['lon'] = result.lon_used
    
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
