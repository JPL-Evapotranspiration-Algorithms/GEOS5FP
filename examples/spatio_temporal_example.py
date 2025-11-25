#!/usr/bin/env python3
"""
Example script demonstrating vectorized multi-variable spatio-temporal queries.

This script shows how to efficiently query multiple GEOS-5 FP variables at different
locations and times by passing lists directly to the variable() method.
"""

import sys
sys.path.insert(0, '..')
import geopandas as gpd
import pandas as pd
from datetime import datetime
from GEOS5FP import GEOS5FPConnection
from spatiotemporal_utils import load_spatiotemporal_csv

# Load spatio_temporal.csv as a GeoDataFrame
print("Loading spatio-temporal data...")
gdf = load_spatiotemporal_csv('../notebooks/spatio_temporal.csv')
print(f"Loaded {len(gdf)} spatio-temporal records")
print(f"Date range: {gdf['time_UTC'].min()} to {gdf['time_UTC'].max()}")
print()

# Test with first 5 records
print("Testing vectorized query with first 5 records...")
gdf_test = gdf.head(5)
print(gdf_test)
print()

# Initialize GEOS-5 FP connection
print("Creating GEOS-5 FP connection...")
conn = GEOS5FPConnection()

# Query multiple variables using vectorized operation
# Use predefined variables from GEOS5FP_VARIABLES
variable_names = ["Ta_K", "SM", "LAI"]
print(f"Querying variables: {variable_names}")
print()

# Vectorized query: pass lists of times and geometries directly
print("Executing vectorized query (no row-by-row iteration)...")
results = conn.query(
    variable_name=variable_names,
    time_UTC=gdf_test['time_UTC'],
    geometry=gdf_test['geometry']
)

# Add ID column from original data
results['ID'] = gdf_test['ID'].values

# Reorder columns: time index, variable columns, ID, geometry
var_cols = [col for col in results.columns if col not in ['ID', 'geometry']]
results = results[var_cols + ['ID', 'geometry']]

print("\nResults:")
print(results)
print()
print(f"Result shape: {results.shape}")
print(f"Result type: {type(results)}")
print(f"CRS: {results.crs}")
print()

# Export to CSV
output_file = '../notebooks/spatio_temporal_results.csv'
print(f"Exporting results to {output_file}...")
results.to_csv(output_file)
print(f"Successfully exported {len(results)} records")
print()

# Now process full dataset
print("=" * 70)
print(f"Processing full dataset ({len(gdf)} records) with vectorized query...")
print("=" * 70)

results_full = conn.query(
    variable_name=variable_names,
    time_UTC=gdf['time_UTC'],
    geometry=gdf['geometry']
)

# Add ID column
results_full['ID'] = gdf['ID'].values

# Reorder columns
var_cols = [col for col in results_full.columns if col not in ['ID', 'geometry']]
results_full = results_full[var_cols + ['ID', 'geometry']]

# Export full results
output_file_full = 'spatio_temporal_results_full.csv'
print(f"\nExporting full results to {output_file_full}...")
results_full.to_csv(output_file_full)
print(f"Successfully exported {len(results_full)} records")

print("\n✓ Vectorized spatio-temporal query completed successfully")
print(f"\nFirst 10 results:")
print(results_full.head(10))
print(f"\nLast 10 results:")
print(results_full.tail(10))

