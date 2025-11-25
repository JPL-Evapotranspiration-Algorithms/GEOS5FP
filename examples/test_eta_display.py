"""
Example demonstrating the ETA display feature during query operations.

This script shows how the query method now displays:
- Elapsed time since query started
- Estimated time remaining (ETA)
- Progress through batches/points

The ETA is calculated based on the average time per completed request
and updates after each server request completes.
"""

from datetime import datetime, timedelta
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
from GEOS5FP import GEOS5FPConnection

# Create connection
conn = GEOS5FPConnection()

print("=" * 80)
print("Example 1: Batch spatio-temporal query with ETA tracking")
print("=" * 80)
print()

# Create a small batch of spatio-temporal queries
# (multiple points at different times)
times = [
    datetime(2024, 11, 1, 12, 0),
    datetime(2024, 11, 2, 12, 0),
    datetime(2024, 11, 3, 12, 0),
    datetime(2024, 11, 4, 12, 0),
    datetime(2024, 11, 5, 12, 0),
]

lats = [34.05, 35.0, 36.0, 37.0, 38.0]
lons = [-118.25, -119.0, -120.0, -121.0, -122.0]

print(f"Querying {len(times)} spatio-temporal points...")
print("You should see ETA updates after the first query completes.")
print()

try:
    result = conn.query(
        target_variables="T2M",
        time_UTC=times,
        lat=lats,
        lon=lons,
        dataset="tavg1_2d_slv_Nx"
    )
    print("\nQuery completed!")
    print(result)
except Exception as e:
    print(f"\nQuery failed (this is expected if you don't have network access): {e}")

print()
print("=" * 80)
print("Example 2: Time-series query with multiple points")
print("=" * 80)
print()

# Create time range query with multiple points
end_time = datetime(2024, 11, 15)
start_time = end_time - timedelta(days=3)

points = [
    Point(-118.25, 34.05),  # Los Angeles
    Point(-122.42, 37.77),  # San Francisco
    Point(-74.00, 40.71),   # New York
]

print(f"Querying {len(points)} points for time range {start_time} to {end_time}...")
print("You should see ETA updates as each point is queried.")
print()

try:
    result = conn.query(
        target_variables="T2M",
        time_range=(start_time, end_time),
        geometry=gpd.GeoSeries(points),
        dataset="tavg1_2d_slv_Nx"
    )
    print("\nQuery completed!")
    print(result)
except Exception as e:
    print(f"\nQuery failed (this is expected if you don't have network access): {e}")

print()
print("=" * 80)
print("Example 3: Multi-variable point query with ETA")
print("=" * 80)
print()

# Query multiple variables at single time for multiple points
print(f"Querying 3 variables at {len(points)} points...")
print("You should see ETA updates for each variable-point combination.")
print()

try:
    result = conn.query(
        target_variables=["T2M", "PS", "QV2M"],
        time_UTC=datetime(2024, 11, 15, 12, 0),
        geometry=gpd.GeoSeries(points),
        dataset="tavg1_2d_slv_Nx"
    )
    print("\nQuery completed!")
    print(result)
except Exception as e:
    print(f"\nQuery failed (this is expected if you don't have network access): {e}")

print()
print("=" * 80)
print("The ETA feature helps track progress for long-running queries!")
print("=" * 80)
