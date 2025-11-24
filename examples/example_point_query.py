"""
Example demonstrating point query functionality for GEOS5FP.

This example shows how to:
1. Query a single point location
2. Query multiple points
3. Get time-series data as a DataFrame
"""

from shapely.geometry import Point, MultiPoint
from datetime import datetime

# Note: Requires xarray and netcdf4 to be installed:
# conda install -c conda-forge xarray netcdf4

try:
    from GEOS5FP import GEOS5FPConnection
    
    # Create connection
    conn = GEOS5FPConnection()
    
    # Example 1: Single point query
    print("=" * 60)
    print("Example 1: Single Point Query")
    print("=" * 60)
    
    # Create a point (lon, lat format for shapely)
    point = Point(-118.25, 34.05)  # Los Angeles
    
    # Use a recent date that should have data
    time_utc = datetime(2024, 11, 15, 12, 0)
    
    # Query surface temperature
    result = conn.Ts_K(time_UTC=time_utc, geometry=point)
    print(f"\nResult type: {type(result)}")
    print(f"\nDataFrame:\n{result}")
    
    # Example 2: Multiple points query
    print("\n" + "=" * 60)
    print("Example 2: Multiple Points Query")
    print("=" * 60)
    
    # Create multiple points
    points = MultiPoint([
        (-118.25, 34.05),   # Los Angeles
        (-122.42, 37.77),   # San Francisco
        (-73.94, 40.73)     # New York
    ])
    
    # Query air temperature
    result = conn.Ta_K(time_UTC=time_utc, geometry=points)
    print(f"\nResult type: {type(result)}")
    print(f"\nDataFrame:\n{result}")
    
    # Example 3: Other variables
    print("\n" + "=" * 60)
    print("Example 3: Querying Different Variables")
    print("=" * 60)
    
    single_point = Point(-118.25, 34.05)
    
    # Surface moisture
    sm_result = conn.SM(time_UTC=time_utc, geometry=single_point)
    print(f"\nSoil Moisture:\n{sm_result}")
    
    # Wind speed components
    u_result = conn.U2M(time_UTC=time_utc, geometry=single_point)
    print(f"\nEastward Wind (U2M):\n{u_result}")
    
    v_result = conn.V2M(time_UTC=time_utc, geometry=single_point)
    print(f"\nNorthward Wind (V2M):\n{v_result}")
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
    
except ImportError as e:
    print(f"ImportError: {e}")
    print("\nTo use point query functionality, install required packages:")
    print("  conda install -c conda-forge xarray netcdf4")
except Exception as e:
    print(f"Error: {e}")
    print("\nNote: This example requires internet connection to query GEOS-5 FP OPeNDAP server")
