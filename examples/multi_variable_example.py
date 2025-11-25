"""
Example script demonstrating multi-variable point queries with GEOS-5 FP

This script shows how to query multiple variables simultaneously at point locations,
with each variable becoming a column in the output DataFrame.
"""

from datetime import datetime, timedelta
from GEOS5FP import GEOS5FPConnection
from shapely.geometry import Point

# Initialize connection
conn = GEOS5FPConnection()

# Define a time range
end_time = datetime(2019, 10, 2, 19, 9, 40)
start_time = end_time - timedelta(hours=6)

# Define a point location (North Carolina)
lat = 35.799
lon = -76.656

print("Example 1: Single variable query")
print("=" * 50)
df_single = conn.query(
    "T2M",
    time_range=(start_time, end_time),
    dataset="tavg1_2d_slv_Nx",
    lat=lat,
    lon=lon
)
print(df_single)
print()

print("Example 2: Multiple variables query (as list)")
print("=" * 50)
df_multi = conn.query(
    ["T2M", "PS", "QV2M"],
    time_range=(start_time, end_time),
    dataset="tavg1_2d_slv_Nx",
    lat=lat,
    lon=lon
)
print(df_multi)
print()

print("Example 3: Multiple predefined variables (no dataset needed)")
print("=" * 50)
df_predefined = conn.query(
    ["Ta_K", "SM", "LAI"],
    time_range=(start_time, end_time),
    lat=lat,
    lon=lon
)
print(df_predefined)
print()

print("Example 4: Single time, multiple variables at point")
print("=" * 50)
df_single_time = conn.query(
    ["T2M", "PS", "QV2M"],
    time_UTC=end_time,
    dataset="tavg1_2d_slv_Nx",
    lat=lat,
    lon=lon
)
print(df_single_time)
