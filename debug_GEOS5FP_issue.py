#!/usr/bin/env python
"""
Debug script to reproduce GEOS5FP timestamp issue.

This script reproduces the exact error from FLiESANN where GEOS5FP.COT() 
is called with a pandas Series for time_UTC instead of individual timestamps.

Error message:
    Failed to query point (29.7381, -82.2188): Cannot convert input [...] 
    of type <class 'pandas.core.series.Series'> to Timestamp
"""

import pandas as pd
import numpy as np
import logging
from GEOS5FP import GEOS5FP
from ECOv002_calval_tables import load_calval_table

# Configure logging to ensure all output is visible
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def main():
    print("=" * 80)
    print("GEOS5FP Debug Script - Reproducing Series Timestamp Error")
    print("=" * 80)
    
    # Load the calibration/validation table
    print("\n1. Loading calibration/validation table...")
    calval_df = load_calval_table()
    print(f"   Loaded {len(calval_df)} rows")
    
    # Ensure `time_UTC` is in datetime format
    calval_df['time_UTC'] = pd.to_datetime(calval_df['time_UTC'])
    
    # Create a `date_UTC` column by extracting the date from `time_UTC`
    calval_df['date_UTC'] = calval_df['time_UTC'].dt.date
    
    # Filter the dataset to only include the first date
    first_date = calval_df['date_UTC'].min()
    calval_df = calval_df[calval_df['date_UTC'] == first_date]
    print(f"   Filtered to first date ({first_date}): {len(calval_df)} rows")
    
    # Extract coordinates and times (mimicking what FLiESANN does)
    print("\n2. Extracting coordinates and times...")
    
    # Handle GeoDataFrame vs regular DataFrame
    if hasattr(calval_df, 'geometry'):
        geometry = calval_df.geometry.values
        print(f"   Geometry type: {type(geometry)}, shape: {geometry.shape}")
        print(f"   Sample geometry: {geometry[0]}")
    elif 'Lat' in calval_df.columns and 'Long' in calval_df.columns:
        latitudes = calval_df['Lat'].values
        longitudes = calval_df['Long'].values
        geometry = (latitudes, longitudes)
        print(f"   Using Lat/Long columns")
        print(f"   Latitudes: {latitudes}")
        print(f"   Longitudes: {longitudes}")
    else:
        raise ValueError(f"Cannot find coordinate columns. Available: {list(calval_df.columns)}")
    
    time_UTC = calval_df['time_UTC']  # This is a pandas Series!
    
    print(f"\n3. Problem setup:")
    print(f"   geometry type: {type(geometry)}")
    print(f"   time_UTC type: {type(time_UTC)}")
    print(f"   time_UTC dtype: {time_UTC.dtype}")
    print(f"   time_UTC values:\n{time_UTC}")
    
    # Initialize GEOS5FP connection
    print("\n4. Initializing GEOS5FP connection...")
    GEOS5FP_connection = GEOS5FP(download_directory="GEOS5FP_download")
    print("   Connection established")
    
    # Reproduce the exact error from FLiESANN
    print("\n5. REPRODUCING THE ERROR...")
    print("   Calling GEOS5FP_connection.COT() with:")
    print(f"     time_UTC = {time_UTC!r}  # <-- pandas Series!")
    print(f"     geometry = {geometry!r}")
    print()
    print("   Executing: GEOS5FP_connection.COT(time_UTC=time_UTC, geometry=geometry)")
    print()
    print("=" * 80)
    
    error_occurred = False
    error_info = None
    
    try:
        COT = GEOS5FP_connection.COT(
            time_UTC=time_UTC,  # This is a pandas Series - THE PROBLEM!
            geometry=geometry
        )
        print(f"\n   UNEXPECTED SUCCESS!")
        print(f"   Result: {COT}")
    except Exception as e:
        error_occurred = True
        error_info = e
        # Let the full traceback print naturally
        import traceback
        print("\nFULL TRACEBACK:")
        print("=" * 80)
        traceback.print_exc()
        print("=" * 80)
    
    # Show the diagnosis
    if error_occurred:
        print("\n" + "=" * 80)
        print("ERROR SUCCESSFULLY REPRODUCED!")
        print("=" * 80)
        print(f"Error Type: {type(error_info).__name__}")
        print(f"Error Message: {error_info}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print("The issue occurs in FLiESANN/retrieve_FLiESANN_GEOS5FP_inputs.py at line 67:")
    print()
    print("    COT = GEOS5FP_connection.COT(")
    print("        time_UTC=time_UTC,      # <-- This is a pandas Series!")
    print("        geometry=geometry,")
    print("        resampling=resampling")
    print("    )")
    print()
    print("When FLiESANN processes data in vectorized mode (row_wise=False),")
    print("it passes the entire time_UTC column (a pandas Series) to GEOS5FP.COT().")
    print()
    print("GEOS5FP.COT() expects either:")
    print("  - A single datetime/Timestamp for point queries")
    print("  - A single datetime/Timestamp for raster queries")
    print()
    print("But it receives a pandas Series with multiple timestamps, which causes:")
    print("  'Cannot convert input [...] of type <class 'pandas.core.series.Series'> to Timestamp'")
    print()
    print("=" * 80)
    print("POSSIBLE SOLUTIONS")
    print("=" * 80)
    print("1. Process row-by-row in FLiESANN:")
    print("   - Set row_wise=True when calling process_FLiESANN_table()")
    print()
    print("2. Fix the vectorized mode in FLiESANN:")
    print("   - Loop through unique time_UTC values")
    print("   - Query GEOS5FP once per unique timestamp")
    print("   - Map results back to the full dataset")
    print()
    print("3. Modify GEOS5FP package:")
    print("   - Handle pandas Series inputs")
    print("   - Process each timestamp separately internally")
    print("=" * 80)


if __name__ == "__main__":
    main()
