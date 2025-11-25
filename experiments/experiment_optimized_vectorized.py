#!/usr/bin/env python3
"""
Test demonstrating the optimized vectorized query with time clustering.
"""

import sys
sys.path.insert(0, '..')
from spatiotemporal_utils import load_spatiotemporal_csv
from GEOS5FP import GEOS5FPConnection
from collections import Counter

# Ensure output is not buffered for live console display
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Load test data
print("=" * 70, flush=True)
print("OPTIMIZED VECTORIZED QUERY TEST (with time clustering)", flush=True)
print("=" * 70, flush=True)
print(flush=True)

gdf = load_spatiotemporal_csv('../notebooks/spatio_temporal.csv').head(20)

# Analyze unique coordinates
unique_coords = gdf.geometry.apply(lambda g: (round(g.x, 6), round(g.y, 6)))
coord_counts = Counter(unique_coords)

print(f"Dataset Analysis:", flush=True)
print(f"  Total records: {len(gdf)}", flush=True)
print(f"  Unique coordinates: {len(coord_counts)}", flush=True)
print(f"  Date range: {gdf['time_UTC'].min()} to {gdf['time_UTC'].max()}", flush=True)
print(flush=True)

print("Records per coordinate:", flush=True)
for coord, count in sorted(coord_counts.items(), key=lambda x: -x[1])[:5]:
    lat, lon = coord
    # Find time range for this coordinate
    coord_mask = unique_coords == coord
    coord_times = gdf[coord_mask]['time_UTC']
    time_span_days = (coord_times.max() - coord_times.min()).total_seconds() / 86400
    print(f"  ({lat:.4f}, {lon:.4f}): {count} records, {time_span_days:.1f} day span", flush=True)
print(flush=True)

# Calculate efficiency
variables = 3
queries_unoptimized = len(gdf) * variables
queries_optimized_estimate = len(coord_counts) * variables  # Lower bound

print("Query Efficiency:", flush=True)
print(f"  Variables queried: {variables}", flush=True)
print(f"  Without optimization: {len(gdf)} records × {variables} vars = {queries_unoptimized} queries", flush=True)
print(f"  With optimization: ~{queries_optimized_estimate} queries (may be more if time clustering needed)", flush=True)
print(f"  Time clustering: keeps each query under 30 days to avoid excessive data transfer", flush=True)
print(flush=True)

# Execute optimized query
print("=" * 70, flush=True)
print("Executing optimized vectorized query with time clustering...", flush=True)
print("=" * 70, flush=True)
print(flush=True)

conn = GEOS5FPConnection()

results = conn.query(
    variable_name=['Ta_K', 'SM', 'LAI'],
    time_UTC=gdf['time_UTC'],
    geometry=gdf['geometry']
)

# Add IDs
results['ID'] = gdf['ID'].values

# Reorder columns
var_cols = [col for col in results.columns if col not in ['ID', 'geometry']]
results = results[var_cols + ['ID', 'geometry']]

print(flush=True)
print("=" * 70, flush=True)
print("RESULTS", flush=True)
print("=" * 70, flush=True)
print(flush=True)
print(results, flush=True)
print(flush=True)
print(f"✅ Successfully retrieved {len(results)} records", flush=True)
print(f"✅ Shape: {results.shape}", flush=True)
print(f"✅ Columns: {list(results.columns)}", flush=True)
print(f"✅ CRS: {results.crs}", flush=True)
