"""
Test FLiESANN inputs script with just the first 3 records
"""

import pandas as pd
from datetime import datetime
from shapely.geometry import Point
from GEOS5FP import GEOS5FPConnection
import time

# Load only first 3 records
df = pd.read_csv('../notebooks/ECOv003_calval_times_locations.csv', nrows=3)
print(f"Testing with {len(df)} records\n")

# Parse coordinates
def parse_point(geometry_str):
    """Extract lon, lat from POINT (lon lat) string"""
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

for idx, row in df.iterrows():
    site_id = row['ID']
    time_utc = row['time_UTC']
    lat = row['lat']
    lon = row['lon']
    
    print(f"Processing {idx+1}/{len(df)}: {site_id} at {time_utc}")
    
    try:
        result_dict = {
            'ID': site_id,
            'time_UTC': time_utc,
            'lat': lat,
            'lon': lon
        }
        
        # Query each variable
        for var_name in ["COT", "AOT", "vapor_gccm", "ozone_cm"]:
            try:
                var_df = conn.query(
                    var_name,
                    time_UTC=time_utc,
                    lat=lat,
                    lon=lon
                )
                
                if len(var_df) > 0 and var_name in var_df.columns:
                    result_dict[var_name] = var_df[var_name].iloc[0]
                    print(f"  {var_name}: {result_dict[var_name]:.4f}")
                else:
                    result_dict[var_name] = None
                    print(f"  {var_name}: No data")
                    
            except Exception as var_e:
                print(f"  {var_name}: Error - {var_e}")
                result_dict[var_name] = None
        
        results.append(pd.DataFrame([result_dict]))
        
    except Exception as e:
        print(f"  Error: {e}")

# Combine results
final_df = pd.concat(results, ignore_index=True)
print("\nResults:")
print(final_df)
