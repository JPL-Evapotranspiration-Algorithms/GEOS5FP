"""
Test multi-variable query with temporal interpolation
"""

from datetime import datetime
from GEOS5FP import GEOS5FPConnection

# Initialize connection
conn = GEOS5FPConnection()

# Test multiple variables at a single point and time
time_utc = datetime(2019, 10, 2, 19, 9, 40)
lat = 35.799
lon = -76.656

print("Testing multi-variable query with temporal_interpolation='interpolate'")
print("=" * 70)
try:
    result = conn.query(
        ["COT", "AOT", "vapor_gccm", "ozone_cm"],
        time_UTC=time_utc,
        lat=lat,
        lon=lon,
        temporal_interpolation="interpolate"
    )
    print("Success!")
    print(f"\nResult shape: {result.shape}")
    print(f"Columns: {result.columns.tolist()}")
    print(f"\nData:")
    print(result)
    
    # Check that all variables are in a single row
    if len(result) == 1:
        print("\n✓ All variables successfully merged into a single row!")
    else:
        print(f"\n⚠ Warning: Expected 1 row, got {len(result)} rows")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Testing multi-variable query with temporal_interpolation='nearest'")
print("=" * 70)
try:
    result = conn.query(
        ["COT", "AOT", "vapor_gccm", "ozone_cm"],
        time_UTC=time_utc,
        lat=lat,
        lon=lon,
        temporal_interpolation="nearest"
    )
    print("Success!")
    print(f"\nResult shape: {result.shape}")
    print(f"Columns: {result.columns.tolist()}")
    print(f"\nData:")
    print(result)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
