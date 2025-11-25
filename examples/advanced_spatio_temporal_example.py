"""
Advanced example: Query multiple GEOS-5 FP variables for multiple spatio-temporal points

This script demonstrates how to:
1. Load a CSV with point locations and timestamps
2. Query multiple meteorological variables for each point
3. Combine results into a comprehensive dataset
"""

import sys
sys.path.insert(0, '..')
import geopandas as gpd
import pandas as pd
from datetime import datetime
from GEOS5FP import GEOS5FPConnection
from spatiotemporal_utils import load_spatiotemporal_csv

# Load spatio-temporal data
print("Loading spatio-temporal.csv...")
gdf = load_spatiotemporal_csv('../notebooks/spatio_temporal.csv')
print(f"Loaded {len(gdf)} records")
print()

# Initialize GEOS-5 FP connection
conn = GEOS5FPConnection()

# Select first 5 records for demonstration
sample_records = gdf.head(5)

print("Sample records:")
print(sample_records[['ID', 'time_UTC', 'geometry']])
print()

# Define variables to query
variables = ["T2M", "PS", "QV2M"]  # Temperature, Pressure, Specific Humidity
dataset = "tavg1_2d_slv_Nx"

print(f"Querying variables: {', '.join(variables)}")
print(f"Dataset: {dataset}")
print("=" * 70)

# Query each location
results = []
for idx, row in sample_records.iterrows():
    lat = row.geometry.y
    lon = row.geometry.x
    time_utc = row['time_UTC']
    station_id = row['ID']
    
    # Convert time string to datetime if needed
    if isinstance(time_utc, str):
        time_utc = datetime.strptime(time_utc, '%Y-%m-%d %H:%M:%S')
    
    print(f"\nQuerying {station_id} at ({lat:.4f}, {lon:.4f}) - {time_utc}")
    
    try:
        # Query multiple variables at this point
        df = conn.query(
            variables,
            time_UTC=time_utc,
            dataset=dataset,
            lat=lat,
            lon=lon
        )
        
        # Add station ID
        df['ID'] = station_id
        df['query_lat'] = lat
        df['query_lon'] = lon
        
        print(f"  Success! Retrieved {len(df)} records")
        results.append(df)
        
    except Exception as e:
        print(f"  Error: {e}")
        continue

# Combine all results
if results:
    print("\n" + "=" * 70)
    print("COMBINED RESULTS")
    print("=" * 70)
    
    final_df = pd.concat(results, ignore_index=False)
    print(final_df)
    
    # Show summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    for var in variables:
        if var in final_df.columns:
            print(f"\n{var}:")
            print(f"  Mean: {final_df[var].mean():.4f}")
            print(f"  Min:  {final_df[var].min():.4f}")
            print(f"  Max:  {final_df[var].max():.4f}")
            print(f"  Std:  {final_df[var].std():.4f}")
    
    # Save to CSV
    output_file = 'geos5fp_query_results.csv'
    final_df.to_csv(output_file)
    print(f"\nResults saved to: {output_file}")
else:
    print("\nNo successful queries")
