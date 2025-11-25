#!/usr/bin/env python3
"""
Simple test of vectorized query with just 2 records.
"""

import geopandas as gpd
import pandas as pd
from datetime import datetime
from shapely.geometry import Point
from GEOS5FP import GEOS5FPConnection

# Create simple test data
data = {
    'ID': ['Test1', 'Test2'],
    'time_UTC': [
        datetime(2019, 10, 2, 19, 0, 0),
        datetime(2019, 10, 3, 19, 0, 0)
    ],
    'geometry': [
        Point(-76.656, 35.799),
        Point(-76.5, 35.8)
    ]
}

gdf = gpd.GeoDataFrame(data, geometry='geometry', crs='EPSG:4326')
print("Test data:")
print(gdf)
print()

# Initialize connection
print("Creating connection...")
conn = GEOS5FPConnection()
print("Connection created")
print()

# Test vectorized query
print("Testing vectorized query...")
print(f"  Variables: ['Ta_K']")
print(f"  Times: {gdf['time_UTC'].tolist()}")
print(f"  Geometries: {gdf['geometry'].tolist()}")
print()

try:
    results = conn.query(
        variable_name="Ta_K",
        time_UTC=gdf['time_UTC'],
        geometry=gdf['geometry']
    )
    
    print("Results:")
    print(results)
    print()
    print(f"Shape: {results.shape}")
    print(f"Type: {type(results)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
