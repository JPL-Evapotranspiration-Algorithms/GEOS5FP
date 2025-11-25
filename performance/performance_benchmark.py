#!/usr/bin/env python3
"""
Performance benchmark comparing different query strategies for GEOS-5 FP data.

Tests three approaches:
1. Naive row-by-row: Query each record individually
2. Optimized vectorized: Use coordinate grouping + time clustering
3. Time-series: Use time_range queries for each coordinate
"""

import sys
import time
import pandas as pd
from datetime import datetime, timedelta
sys.path.insert(0, '..')
from spatiotemporal_utils import load_spatiotemporal_csv
from GEOS5FP import GEOS5FPConnection

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def benchmark_naive_row_by_row(gdf, variable_names):
    """
    Strategy 1: Query each record individually (naive approach).
    Makes 1 query per record per variable.
    """
    print("\n" + "=" * 70)
    print("STRATEGY 1: NAIVE ROW-BY-ROW")
    print("=" * 70)
    print(f"Approach: Query each record individually")
    print(f"Expected queries: {len(gdf)} records × {len(variable_names)} vars = {len(gdf) * len(variable_names)}")
    print()
    
    conn = GEOS5FPConnection()
    start_time = time.time()
    
    results = []
    query_count = 0
    
    for idx, row in gdf.iterrows():
        for var_name in variable_names:
            query_count += 1
            try:
                # Query single time point
                result = conn.query(
                    variable_name=var_name,
                    time_UTC=row['time_UTC'],
                    geometry=row['geometry']
                )
                
                if idx == 0:  # Only print for first record
                    print(f"  Query {query_count}: {var_name} at {row['geometry']} on {row['time_UTC']}")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        if idx == 0:
            print(f"  ... continuing for {len(gdf)-1} more records ...")
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Completed {query_count} queries in {elapsed:.1f} seconds")
    print(f"  Average: {elapsed/query_count:.2f} seconds per query")
    print(f"  Throughput: {query_count/elapsed:.2f} queries/second")
    
    return {
        'strategy': 'naive_row_by_row',
        'queries': query_count,
        'time': elapsed,
        'avg_query_time': elapsed/query_count,
        'queries_per_sec': query_count/elapsed
    }


def benchmark_optimized_vectorized(gdf, variable_names):
    """
    Strategy 2: Optimized vectorized with coordinate grouping + time clustering.
    Makes 1 query per coordinate-time cluster per variable.
    """
    print("\n" + "=" * 70)
    print("STRATEGY 2: OPTIMIZED VECTORIZED (Coordinate + Time Clustering)")
    print("=" * 70)
    print(f"Approach: Group by coordinates, cluster times (max 30 days)")
    print()
    
    conn = GEOS5FPConnection()
    start_time = time.time()
    
    try:
        results = conn.query(
            variable_name=variable_names,
            time_UTC=gdf['time_UTC'],
            geometry=gdf['geometry']
        )
        
        elapsed = time.time() - start_time
        
        # Extract actual query count from implementation
        # This is approximate based on the batches shown in logs
        print(f"\n✓ Completed vectorized query in {elapsed:.1f} seconds")
        
        return {
            'strategy': 'optimized_vectorized',
            'queries': None,  # Filled from logs
            'time': elapsed,
            'avg_query_time': None,
            'queries_per_sec': None
        }
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_time_series(gdf, variable_names):
    """
    Strategy 3: Query full time-series for each unique coordinate.
    Makes 1 query per coordinate per variable (no time clustering).
    """
    print("\n" + "=" * 70)
    print("STRATEGY 3: TIME-SERIES (Full range per coordinate)")
    print("=" * 70)
    
    # Group by unique coordinates
    from collections import defaultdict
    coord_to_records = defaultdict(list)
    
    for idx, row in gdf.iterrows():
        coord_key = (round(row.geometry.y, 6), round(row.geometry.x, 6))
        coord_to_records[coord_key].append({
            'index': idx,
            'time': row['time_UTC'],
            'geometry': row['geometry'],
            'ID': row['ID']
        })
    
    unique_coords = len(coord_to_records)
    expected_queries = unique_coords * len(variable_names)
    
    print(f"Approach: Query full time range for each coordinate")
    print(f"Unique coordinates: {unique_coords}")
    print(f"Expected queries: {unique_coords} coords × {len(variable_names)} vars = {expected_queries}")
    
    # Calculate total time span and average span per coordinate
    total_span_days = 0
    for coord_key, records in coord_to_records.items():
        times = [r['time'] for r in records]
        span = (max(times) - min(times)).total_seconds() / 86400
        total_span_days += span
    avg_span = total_span_days / unique_coords
    
    print(f"Average time span per coordinate: {avg_span:.1f} days")
    print()
    
    conn = GEOS5FPConnection()
    start_time = time.time()
    
    results_by_index = {}
    query_count = 0
    
    for coord_idx, (coord_key, records) in enumerate(coord_to_records.items(), 1):
        lat, lon = coord_key
        
        # Get time range for this coordinate
        times = [r['time'] for r in records]
        min_time = min(times)
        max_time = max(times)
        time_span_days = (max_time - min_time).total_seconds() / 86400
        
        if coord_idx <= 3:  # Print first 3 coordinates
            print(f"  Coord {coord_idx}/{unique_coords}: ({lat:.4f}, {lon:.4f}) - "
                  f"{len(records)} records, {time_span_days:.1f} day span")
        
        # Query each variable for this coordinate's full time range
        for var_name in variable_names:
            query_count += 1
            
            try:
                # Add buffer
                time_range_start = min_time - timedelta(hours=2)
                time_range_end = max_time + timedelta(hours=2)
                
                # Query using time_range
                from GEOS5FP.constants import GEOS5FP_VARIABLES
                if var_name in GEOS5FP_VARIABLES:
                    from GEOS5FP.GEOS5FP_point import query_geos5fp_point
                    _, dataset, raw_variable = conn._get_variable_info(var_name)
                    variable_opendap = raw_variable.lower()
                    
                    result = query_geos5fp_point(
                        dataset=dataset,
                        variable=variable_opendap,
                        lat=lat,
                        lon=lon,
                        time_range=(time_range_start, time_range_end),
                        dropna=True
                    )
                    
                    # Extract specific times needed
                    for record in records:
                        if record['index'] not in results_by_index:
                            results_by_index[record['index']] = {
                                'time_UTC': record['time'],
                                'geometry': record['geometry'],
                                'ID': record['ID']
                            }
                        
                        # Find closest time
                        time_diffs = abs(result.df.index - record['time'])
                        closest_idx = time_diffs.argmin()
                        value = result.df.iloc[closest_idx][variable_opendap]
                        results_by_index[record['index']][var_name] = value
                
            except Exception as e:
                if coord_idx <= 3:
                    print(f"    Error querying {var_name}: {e}")
        
        if coord_idx == 3 and unique_coords > 3:
            print(f"  ... processing {unique_coords - 3} more coordinates ...")
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Completed {query_count} queries in {elapsed:.1f} seconds")
    print(f"  Average: {elapsed/query_count:.2f} seconds per query")
    print(f"  Throughput: {query_count/elapsed:.2f} queries/second")
    
    return {
        'strategy': 'time_series_full_range',
        'queries': query_count,
        'time': elapsed,
        'avg_query_time': elapsed/query_count,
        'queries_per_sec': query_count/elapsed,
        'avg_span_days': avg_span
    }


def main():
    print("=" * 70, flush=True)
    print("GEOS-5 FP QUERY PERFORMANCE BENCHMARK", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    
    # Load test data
    print("Loading test data...", flush=True)
    gdf = load_spatiotemporal_csv('../notebooks/spatio_temporal.csv')
    
    # Use small subset for quick testing (scales to 1-2 minutes)
    test_size = 5  # 5 records for quick comparison
    gdf_test = gdf.head(test_size)
    
    print(f"Dataset: {len(gdf_test)} records", flush=True)
    print(f"Date range: {gdf_test['time_UTC'].min()} to {gdf_test['time_UTC'].max()}", flush=True)
    
    # Count unique coordinates
    unique_coords = gdf_test.geometry.apply(lambda g: (round(g.x, 6), round(g.y, 6))).nunique()
    print(f"Unique coordinates: {unique_coords}", flush=True)
    
    # Variables to query
    variable_names = ["Ta_K"]  # Single variable for speed
    print(f"Variables: {variable_names}", flush=True)
    print(f"Total queries needed (naive): {len(gdf_test) * len(variable_names)}", flush=True)
    print(flush=True)
    
    # Run benchmarks
    results = []
    
    # Strategy 1: Naive row-by-row
    print("NOTE: All strategies tested with small dataset for 1-2 minute runtime", flush=True)
    print(flush=True)
    
    result1 = benchmark_naive_row_by_row(gdf_test, variable_names)
    if result1:
        results.append(result1)
    
    # Strategy 2: Optimized vectorized
    result2 = benchmark_optimized_vectorized(gdf_test, variable_names)
    if result2:
        results.append(result2)
    
    # Strategy 3: Time-series
    result3 = benchmark_time_series(gdf_test, variable_names)
    if result3:
        results.append(result3)
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 70)
    print()
    
    if results:
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
        print()
        
        # Find fastest
        if len([r for r in results if r['time'] is not None]) > 1:
            fastest = min([r for r in results if r['time'] is not None], key=lambda x: x['time'])
            print(f"🏆 Fastest: {fastest['strategy']} ({fastest['time']:.1f}s)")
            
            # Calculate speedups
            for result in results:
                if result['time'] and result['strategy'] != fastest['strategy']:
                    speedup = result['time'] / fastest['time']
                    print(f"   vs {result['strategy']}: {speedup:.2f}x faster")
            
            print()
            print("EXTRAPOLATION TO FULL DATASET (1,065 records):")
            print("-" * 70)
            
            # Extrapolate to full dataset
            full_size = 1065
            scale_factor = full_size / len(gdf_test)
            
            for result in results:
                if result['time']:
                    if result['queries']:
                        estimated_queries = int(result['queries'] * scale_factor)
                        estimated_time = result['time'] * scale_factor
                        print(f"{result['strategy']}:")
                        print(f"  Estimated queries: {estimated_queries}")
                        print(f"  Estimated time: {estimated_time:.0f}s ({estimated_time/60:.1f} min)")
                    else:
                        # For vectorized, estimate based on coordinate scaling
                        estimated_time = result['time'] * scale_factor
                        print(f"{result['strategy']}:")
                        print(f"  Estimated time: {estimated_time:.0f}s ({estimated_time/60:.1f} min)")
                    print()
    
    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
