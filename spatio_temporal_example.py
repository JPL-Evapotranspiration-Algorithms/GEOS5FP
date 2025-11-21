import geopandas as gpd
import pandas as pd
from datetime import datetime
from GEOS5FP import GEOS5FPConnection
from spatiotemporal_utils import load_spatiotemporal_csv

# Load spatio_temporal.csv as a GeoDataFrame
gdf = load_spatiotemporal_csv('spatio_temporal.csv')

# Uncomment to test with a small sample first:
# gdf = gdf.head(5)

# Print the GeoDataFrame
print("Loaded GeoDataFrame:")
print(gdf.head(10))
print(f"\nTotal records: {len(gdf)}")
print()

# Example: Query multiple GEOS-5 FP variables for all locations
print("=" * 70)
print("Example: Querying multiple GEOS-5 FP variables for all points")
print("=" * 70)

# Initialize GEOS-5 FP connection
conn = GEOS5FPConnection()

# Query multiple variables for all points
print(f"Querying variables: Ta_K, SM, LAI")
print(f"Processing {len(gdf)} records...")
print()

# Use the entire GeoDataFrame's geometry and time columns
# We'll query each record individually and collect results
results = []

for idx, row in gdf.iterrows():
    geometry = row.geometry
    time_utc = row['time_UTC']
    station_id = row['ID']
    
    print(f"Processing {idx + 1}/{len(gdf)}: {station_id} at {time_utc}")
    
    try:
        df = conn.variable(
            ["Ta_K", "SM", "LAI"],
            time_UTC=time_utc,
            geometry=geometry
        )
        
        # Add station ID to the results
        df['ID'] = station_id
        results.append(df)
        
    except Exception as e:
        print(f"  Error: {e}")
        continue

# Combine all results
if results:
    print("\n" + "=" * 70)
    print("COMBINED RESULTS")
    print("=" * 70)
    
    final_gdf = pd.concat(results, ignore_index=False)
    final_gdf = gpd.GeoDataFrame(final_gdf, geometry='geometry', crs='EPSG:4326')
    
    print(f"\nType: {type(final_gdf)}")
    print(f"CRS: {final_gdf.crs}")
    print(f"Total records retrieved: {len(final_gdf)}")
    print("\nFirst 10 records:")
    print(final_gdf.head(10))
    print("\nLast 10 records:")
    print(final_gdf.tail(10))
    
    # Save to file
    output_file = 'geos5fp_results.csv'
    final_gdf.to_csv(output_file, index=True)
    print(f"\nResults saved to: {output_file}")
else:
    print("\nNo successful queries")
