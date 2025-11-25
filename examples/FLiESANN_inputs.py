"""
FLiESANN Inputs Script

This script loads the ECOv003_calval_times_locations.csv table and queries
GEOS-5 FP variables (COT, AOT, vapor_gccm, and ozone_cm) at those times and locations.
"""

import pandas as pd
from datetime import datetime
from shapely.geometry import Point
from GEOS5FP import GEOS5FPConnection
import time

# Load the CSV file
print("Loading ECOv003_calval_times_locations.csv...")
df = pd.read_csv('notebooks/ECOv003_calval_times_locations.csv')
print(f"Loaded {len(df)} records\n")

# Display first few rows
print("First few rows of input data:")
print(df.head())
print()

# Parse the geometry column to extract lat/lon
print("Parsing coordinates from geometry column...")
def parse_point(geometry_str):
    """Extract lon, lat from POINT (lon lat) string"""
    coords = geometry_str.replace('POINT (', '').replace(')', '').split()
    lon = float(coords[0])
    lat = float(coords[1])
    return lon, lat

df[['lon', 'lat']] = df['geometry'].apply(lambda x: pd.Series(parse_point(x)))

# Convert time_UTC to datetime
df['time_UTC'] = pd.to_datetime(df['time_UTC'])

print(f"Date range: {df['time_UTC'].min()} to {df['time_UTC'].max()}")
print(f"Spatial extent: Lat [{df['lat'].min():.3f}, {df['lat'].max():.3f}], Lon [{df['lon'].min():.3f}, {df['lon'].max():.3f}]")
print()

# Initialize GEOS-5 FP connection
print("Initializing GEOS-5 FP connection...")
conn = GEOS5FPConnection()
print()

# Query variables for each location and time
print("Querying GEOS-5 FP variables: COT, AOT, vapor_gccm, ozone_cm")
print("This may take a while...")
print()

# List to store results
results = []

# Timing variables
start_time = time.time()
query_times = []

# Loop through each row and query the variables
for idx, row in df.iterrows():
    site_id = row['ID']
    time_utc = row['time_UTC']
    lat = row['lat']
    lon = row['lon']
    
    # Start timing this query
    query_start = time.time()
    
    # Calculate estimated time remaining
    if idx > 0:
        avg_time_per_query = sum(query_times) / len(query_times)
        remaining_queries = len(df) - idx
        estimated_seconds_remaining = avg_time_per_query * remaining_queries
        
        # Format time remaining
        if estimated_seconds_remaining < 60:
            time_str = f"{estimated_seconds_remaining:.1f}s"
        elif estimated_seconds_remaining < 3600:
            minutes = int(estimated_seconds_remaining // 60)
            seconds = int(estimated_seconds_remaining % 60)
            time_str = f"{minutes}m {seconds}s"
        else:
            hours = int(estimated_seconds_remaining // 3600)
            minutes = int((estimated_seconds_remaining % 3600) // 60)
            time_str = f"{hours}h {minutes}m"
        
        print(f"Processing {idx+1}/{len(df)}: {site_id} at {time_utc} (Lat: {lat:.4f}, Lon: {lon:.4f}) [ETA: {time_str}]")
    else:
        print(f"Processing {idx+1}/{len(df)}: {site_id} at {time_utc} (Lat: {lat:.4f}, Lon: {lon:.4f})")
    
    try:
        # Query all four variables at once using temporal interpolation
        # This handles different temporal resolutions by interpolating to the target time
        result_df = conn.query(
            ["COT", "AOT", "vapor_gccm", "ozone_cm"],
            time_UTC=time_utc,
            lat=lat,
            lon=lon,
            temporal_interpolation="interpolate"
        )
        
        # Extract values from the result
        result_dict = {
            'ID': site_id,
            'time_UTC': time_utc,
            'lat': lat,
            'lon': lon
        }
        
        # Get values for each variable (should be in a single row now)
        if len(result_df) > 0:
            for var_name in ["COT", "AOT", "vapor_gccm", "ozone_cm"]:
                if var_name in result_df.columns:
                    result_dict[var_name] = result_df[var_name].iloc[0]
                else:
                    result_dict[var_name] = None
        else:
            # No data returned
            for var_name in ["COT", "AOT", "vapor_gccm", "ozone_cm"]:
                result_dict[var_name] = None
        
        results.append(pd.DataFrame([result_dict]))
        
    except Exception as e:
        print(f"  Error: {e}")
        # Add a row with NaN values if query fails
        results.append(pd.DataFrame({
            'ID': [site_id],
            'time_UTC': [time_utc],
            'lat': [lat],
            'lon': [lon],
            'COT': [None],
            'AOT': [None],
            'vapor_gccm': [None],
            'ozone_cm': [None]
        }))
    
    # Record query time
    query_end = time.time()
    query_times.append(query_end - query_start)

# Calculate total elapsed time
total_time = time.time() - start_time
if total_time < 60:
    total_time_str = f"{total_time:.1f}s"
elif total_time < 3600:
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    total_time_str = f"{minutes}m {seconds}s"
else:
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    total_time_str = f"{hours}h {minutes}m"

print()
print(f"All queries completed in {total_time_str}")
print()
print("Combining results...")
# Combine all results into a single DataFrame
final_df = pd.concat(results, ignore_index=True)

# Reorder columns for better readability
column_order = ['ID', 'time_UTC', 'lat', 'lon', 'COT', 'AOT', 'vapor_gccm', 'ozone_cm']
final_df = final_df[column_order]

# Save results to CSV
output_file = 'notebooks/FLiESANN_inputs_results.csv'
print(f"Saving results to {output_file}...")
final_df.to_csv(output_file, index=False)

print()
print("Summary Statistics:")
print("=" * 60)
print(final_df[['COT', 'AOT', 'vapor_gccm', 'ozone_cm']].describe())
print()

print("First few results:")
print(final_df.head(10))
print()

print(f"Complete! Results saved to {output_file}")
print(f"Total records processed: {len(final_df)}")
