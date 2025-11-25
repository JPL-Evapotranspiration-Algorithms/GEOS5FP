"""
Test multi-variable query to diagnose the issue
"""

from datetime import datetime
from GEOS5FP import GEOS5FPConnection

# Initialize connection
conn = GEOS5FPConnection()

# Test multiple variables at a single point and time
time_utc = datetime(2019, 10, 2, 19, 9, 40)
lat = 35.799
lon = -76.656

print("Testing multi-variable query (COT, AOT, vapor_gccm, ozone_cm)...")
try:
    result = conn.query(
        ["COT", "AOT", "vapor_gccm", "ozone_cm"],
        time_UTC=time_utc,
        lat=lat,
        lon=lon
    )
    print("Success!")
    print(result)
    print(f"\nColumns: {result.columns.tolist()}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
