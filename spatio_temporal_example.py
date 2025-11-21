import geopandas as gpd
import pandas as pd
from datetime import datetime
from GEOS5FP import GEOS5FPConnection
from spatiotemporal_utils import load_spatiotemporal_csv

# Load spatio_temporal.csv as a GeoDataFrame
gdf = load_spatiotemporal_csv('spatio_temporal.csv')

# Print the GeoDataFrame
print("Loaded GeoDataFrame:")
print(gdf.head(10))
print(f"\nTotal records: {len(gdf)}")
print()

# Example: Query multiple GEOS-5 FP variables for the first location
print("=" * 70)
print("Example: Querying multiple GEOS-5 FP variables at first point")
print("=" * 70)

# Get first record
first_record = gdf.iloc[0]
geometry = first_record.geometry
time_utc = first_record['time_UTC']

print(f"Location: {geometry}")
print(f"Time: {time_utc}")
print()

# Initialize GEOS-5 FP connection
conn = GEOS5FPConnection()

# Query multiple variables at this point
print("Querying variables: Ta_K, SM, LAI")
df = conn.variable(
    ["Ta_K", "SM", "LAI"],
    time_UTC=time_utc,
    geometry=geometry
)

print("\nResults:")
print(f"Type: {type(df)}")
print(f"CRS: {df.crs}")
print(df)
