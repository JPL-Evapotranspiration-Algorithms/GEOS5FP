#!/usr/bin/env python3
"""
Example demonstrating the consolidated variable retrieval system with time-series and flexible geometry support.
"""

from datetime import datetime, timedelta
from GEOS5FP import GEOS5FPConnection
from shapely.geometry import Point, MultiPoint
import pandas as pd
import numpy as np

def main():
    # Create GEOS-5 FP connection
    geos5fp = GEOS5FPConnection()
    
    print("=== GEOS-5 FP Consolidated Variable Retrieval Examples ===\n")
    
    # Example 1: Single time, single raster (backward compatibility)
    print("1. Single time, raster geometry (backward compatible):")
    time_single = "2023-01-01 12:00"
    try:
        sfmc_raster = geos5fp.SFMC(time_single)
        print(f"   SFMC raster shape: {sfmc_raster.shape}")
        print(f"   Type: {type(sfmc_raster)}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Example 2: Single time, point extraction
    print("2. Single time, point extraction:")
    points = [Point(-118.25, 34.05), Point(-74.01, 40.71)]  # LA and NYC
    try:
        sfmc_points = geos5fp.SFMC(time_single, geometry=points)
        print(f"   SFMC at points: {sfmc_points}")
        print(f"   Type: {type(sfmc_points)}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Example 3: Time-series with raster
    print("3. Time-series with raster geometry:")
    times = [
        "2023-01-01 12:00",
        "2023-01-02 12:00", 
        "2023-01-03 12:00"
    ]
    try:
        sfmc_timeseries = geos5fp.SFMC(times)
        print(f"   Time-series DataFrame shape: {sfmc_timeseries.shape}")
        print(f"   Index: {sfmc_timeseries.index.tolist()}")
        print(f"   Columns: {sfmc_timeseries.columns.tolist()}")
        print(f"   Type: {type(sfmc_timeseries)}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Example 4: Time-series with point extraction
    print("4. Time-series with point extraction:")
    try:
        sfmc_points_timeseries = geos5fp.SFMC(times, geometry=points)
        print(f"   Time-series DataFrame shape: {sfmc_points_timeseries.shape}")
        print(f"   Index: {sfmc_points_timeseries.index.tolist()}")
        print(f"   Columns: {sfmc_points_timeseries.columns.tolist()}")
        print(f"   Sample values:")
        print(sfmc_points_timeseries.head())
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Example 5: Multi-variable time-series
    print("5. Multi-variable time-series:")
    try:
        multi_var = geos5fp.get_time_series(
            variable_names=['SFMC', 'LAI', 'TS'],
            time_UTC=times,
            geometry=points
        )
        print(f"   Multi-variable DataFrame shape: {multi_var.shape}")
        print(f"   Columns: {multi_var.columns.tolist()}")
        print(f"   Sample values:")
        print(multi_var.head())
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Example 6: Using coordinate tuples instead of Point objects
    print("6. Using coordinate tuples:")
    coord_points = [(-118.25, 34.05), (-74.01, 40.71)]  # Same locations as tuples
    try:
        lai_coords = geos5fp.LAI(time_single, geometry=coord_points)
        print(f"   LAI at coordinate tuples: {lai_coords}")
        print(f"   Type: {type(lai_coords)}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Example 7: Using MultiPoint geometry
    print("7. Using MultiPoint geometry:")
    try:
        multipoint = MultiPoint([(-118.25, 34.05), (-74.01, 40.71)])
        lai_multipoint = geos5fp.LAI(time_single, geometry=multipoint)
        print(f"   LAI at MultiPoint: {lai_multipoint}")
        print(f"   Type: {type(lai_multipoint)}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Example 8: Available variables
    print("8. Available variables in registry:")
    print(f"   Variables: {list(geos5fp.get_variable.__globals__['GEOS5FP_VARIABLES'].keys())}")
    print()

if __name__ == "__main__":
    main()