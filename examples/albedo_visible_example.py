"""
Example demonstrating the use of the albedo_visible computed variable.

The albedo_visible variable is computed as ALBEDO * (ALBVISDR / ALBEDO), which
simplifies to ALBVISDR - the visible component of surface albedo.

This represents the direct visible beam albedo scaled from GEOS-5 FP products.
"""

from GEOS5FP import GEOS5FP
import logging

# Set up logging to see progress
logging.basicConfig(level=logging.INFO)

# Create GEOS5FP connection
geos5fp = GEOS5FP()

# Example 1: Get albedo_visible for a specific time and location (raster query)
print("Example 1: Raster query for albedo_visible")
try:
    from sentinel_tiles import sentinel_tiles
    geometry = sentinel_tiles.grid("11SPS")
    timestamp = "2025-02-22 12:00:00"
    
    albedo_visible = geos5fp.albedo_visible(time_UTC=timestamp, geometry=geometry)
    print(f"albedo_visible raster: {albedo_visible}")
except ImportError:
    print("Skipping raster example (requires sentinel_tiles)")

# Example 2: Point query for albedo_visible
print("\nExample 2: Point query for albedo_visible")
from datetime import datetime
from shapely.geometry import Point

# Define a point location (longitude, latitude)
point = Point(-118.5, 34.2)  # Los Angeles area
time_UTC = datetime(2025, 2, 22, 12, 0, 0)

# Query albedo_visible for this point
result = geos5fp.albedo_visible(time_UTC=time_UTC, geometry=point)
print(f"albedo_visible at point: {result}")

# Example 3: Multi-point query using the query method
print("\nExample 3: Multi-point query for albedo_visible")
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

# Query albedo_visible for all points
results = geos5fp.query(
    target_variables=['albedo_visible', 'ALBEDO', 'ALBVISDR'],
    targets_df=targets_gdf,
    verbose=True
)

print("\nResults:")
print(results[['time_UTC', 'lon', 'lat', 'ALBEDO', 'ALBVISDR', 'albedo_visible']])

# Verify the computation
print("\nVerification (albedo_visible should equal ALBEDO * (ALBVISDR / ALBEDO) = ALBVISDR):")
computed_value = results['ALBEDO'] * (results['ALBVISDR'] / results['ALBEDO'])
print(f"Computed value: {computed_value.values}")
print(f"albedo_visible: {results['albedo_visible'].values}")
print(f"ALBVISDR: {results['ALBVISDR'].values}")
print(f"Match with ALBVISDR: {all(abs(results['ALBVISDR'] - results['albedo_visible']) < 1e-6)}")

# Interpret the values
print("\nInterpretation:")
for idx, row in results.iterrows():
    ratio = row['albedo_visible'] / row['ALBEDO']
    print(f"  {row['time_UTC']}: albedo_visible = {row['albedo_visible']:.3f}, ALBEDO = {row['ALBEDO']:.3f}, ratio = {ratio:.3f}")
