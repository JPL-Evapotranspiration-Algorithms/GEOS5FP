"""
Test a single query to verify connection works
"""

from datetime import datetime
from GEOS5FP import GEOS5FPConnection

# Initialize connection
conn = GEOS5FPConnection()

# Test a single variable at a single point and time
time_utc = datetime(2019, 10, 2, 19, 9, 40)
lat = 35.799
lon = -76.656

print("Testing single variable query (COT)...")
try:
    result = conn.query(
        "COT",
        time_UTC=time_utc,
        lat=lat,
        lon=lon
    )
    print("Success!")
    print(result)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting another single variable (AOT)...")
try:
    result = conn.query(
        "AOT",
        time_UTC=time_utc,
        lat=lat,
        lon=lon
    )
    print("Success!")
    print(result)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
