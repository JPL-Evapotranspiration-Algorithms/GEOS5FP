"""
Example demonstrating the use of the albedo_NIR computed variable.

The albedo_NIR variable is computed as ALBEDO * (ALBNIRDR / ALBEDO), which
simplifies to ALBNIRDR - the NIR component of surface albedo.

This represents the direct NIR beam albedo scaled from GEOS-5 FP products.
"""

from GEOS5FP import GEOS5FP
import logging

# Set up logging to see progress
logging.basicConfig(level=logging.INFO)

# Create GEOS5FP connection
geos5fp = GEOS5FP()

# Example 1: Get albedo_NIR for a specific time and location (raster query)
print("Example 1: Raster query for albedo_NIR")
try:
    from sentinel_tiles import sentinel_tiles
    geometry = sentinel_tiles.grid("11SPS")
    timestamp = "2025-02-22 12:00:00"
    
    albedo_NIR = geos5fp.albedo_NIR(time_UTC=timestamp, geometry=geometry)
    print(f"albedo_NIR raster: {albedo_NIR}")
except ImportError:
    print("Skipping raster example (requires sentinel_tiles)")

# Example 2: Point query for albedo_NIR
print("\nExample 2: Point query for albedo_NIR")
from datetime import datetime
from shapely.geometry import Point

# Define a point location (longitude, latitude)
point = Point(-118.5, 34.2)  # Los Angeles area
time_UTC = datetime(2025, 2, 22, 12, 0, 0)

# Query albedo_NIR for this point
result = geos5fp.albedo_NIR(time_UTC=time_UTC, geometry=point)
print(f"albedo_NIR at point: {result}")

# Example 3: Multi-point query using the query method
print("\nExample 3: Multi-point query for albedo_NIR")
import pandas as pd
import geopandas as gpd

# Create a DataFrame with multiple locations and times
targets_df = pd.DataFrame({
    'time_UTC': [
        datetime(2025, 2, 22, 12, 0, 0),
        datetime(2025, 2, 23, 12, 0, 0),
        datetime(2025, 2, 24, 12, 0, 0),
    ],
    'lon': [-118.5, -119.0, -117.5],
    'lat': [34.2, 34.5, 33.8]
})

# Add geometry column
targets_df['geometry'] = [Point(lon, lat) for lon, lat in zip(targets_df['lon'], targets_df['lat'])]
targets_gdf = gpd.GeoDataFrame(targets_df, geometry='geometry', crs='EPSG:4326')

# Query albedo_NIR for all points
results = geos5fp.query(
    target_variables=['albedo_NIR', 'ALBEDO', 'ALBNIRDR'],
    targets_df=targets_gdf,
    verbose=True
)

print("\nResults:")
print(results[['time_UTC', 'lon', 'lat', 'ALBEDO', 'ALBNIRDR', 'albedo_NIR']])

# Verify the computation
print("\nVerification (albedo_NIR should equal ALBEDO * (ALBNIRDR / ALBEDO) = ALBNIRDR):")
computed_value = results['ALBEDO'] * (results['ALBNIRDR'] / results['ALBEDO'])
print(f"Computed value: {computed_value.values}")
print(f"albedo_NIR: {results['albedo_NIR'].values}")
print(f"ALBNIRDR: {results['ALBNIRDR'].values}")
print(f"Match with ALBNIRDR: {all(abs(results['ALBNIRDR'] - results['albedo_NIR']) < 1e-6)}")

# Example 4: Query both visible and NIR albedo together
print("\nExample 4: Query both albedo_visible and albedo_NIR")
results_both = geos5fp.query(
    target_variables=['albedo_visible', 'albedo_NIR', 'ALBEDO'],
    targets_df=targets_gdf,
    verbose=True
)

print("\nResults with all albedo types:")
print(results_both[['time_UTC', 'lon', 'lat', 'albedo_visible', 'albedo_NIR', 'ALBEDO']])
