"""
Test FLiESANN inputs script with temporal interpolation - first 5 records
"""

import pandas as pd
from datetime import datetime
from shapely.geometry import Point
from GEOS5FP import GEOS5FPConnection
import time

# Load only first 5 records
df = pd.read_csv('../notebooks/ECOv003_calval_times_locations.csv', nrows=5)
print(f"Testing with {len(df)} records\n")

# Parse coordinates
def parse_point(geometry_str):
    coords = geometry_str.replace('POINT (', '').replace(')', '').split()
    lon = float(coords[0])
    lat = float(coords[1])
    return lon, lat

df[['lon', 'lat']] = df['geometry'].apply(lambda x: pd.Series(parse_point(x)))
df['time_UTC'] = pd.to_datetime(df['time_UTC'])

# Initialize connection
conn = GEOS5FPConnection()

# Query variables
results = []
start_time = time.time()

for idx, row in df.iterrows():
    site_id = row['ID']
    time_utc = row['time_UTC']
    lat = row['lat']
    lon = row['lon']
    
    print(f"\nProcessing {idx+1}/{len(df)}: {site_id} at {time_utc}")
    
    try:
        # Query all four variables at once using temporal interpolation
        result_df = conn.query(
            ["COT", "AOT", "vapor_gccm", "ozone_cm"],
            time_UTC=time_utc,
            lat=lat,
            lon=lon,
            temporal_interpolation="interpolate"
        )
        
        # Extract values
        result_dict = {
            'ID': site_id,
            'time_UTC': time_utc,
            'lat': lat,
            'lon': lon
        }
        
        if len(result_df) > 0:
            for var_name in ["COT", "AOT", "vapor_gccm", "ozone_cm"]:
                if var_name in result_df.columns:
                    value = result_df[var_name].iloc[0]
                    result_dict[var_name] = value
                    print(f"  {var_name}: {value:.4f}")
                else:
                    result_dict[var_name] = None
                    print(f"  {var_name}: Missing")
        else:
            for var_name in ["COT", "AOT", "vapor_gccm", "ozone_cm"]:
                result_dict[var_name] = None
            print("  No data returned")
        
        results.append(pd.DataFrame([result_dict]))
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

elapsed = time.time() - start_time
print(f"\n\nCompleted {len(results)} queries in {elapsed:.1f}s ({elapsed/len(results):.1f}s per query)")

# Combine results
final_df = pd.concat(results, ignore_index=True)
print("\nResults:")
print(final_df)
