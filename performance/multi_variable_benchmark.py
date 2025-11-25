#!/usr/bin/env python3
"""
Multi-variable query performance benchmark.

Tests the speedup gained by querying multiple variables from the same dataset
in a single request vs sequential queries.
"""

import sys
import time
import pandas as pd
sys.path.insert(0, '..')
from spatiotemporal_utils import load_spatiotemporal_csv
from GEOS5FP import GEOS5FPConnection

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def benchmark_multi_variable(gdf, variable_names):
    """
    Benchmark the optimized vectorized approach with multi-variable support.
    All variables from the same dataset are queried in a single request.
    """
    print(f"\nQuerying {len(variable_names)} variables: {', '.join(variable_names)}")
    print(f"Processing {len(gdf)} records...")
    print()
    
    conn = GEOS5FPConnection()
    start_time = time.time()
    
    try:
        # Use optimized vectorized query (now with multi-variable support)
        result_gdf = conn.query(
            variable_name=variable_names,
            time_UTC=gdf['time_UTC'],
            geometry=gdf['geometry']
        )
        
        elapsed = time.time() - start_time
        
        print(f"✓ Completed in {elapsed:.1f} seconds")
        print(f"  Retrieved {len(result_gdf)} records × {len(variable_names)} variables")
        print(f"  Throughput: {len(result_gdf) * len(variable_names) / elapsed:.1f} values/second")
        
        return {
            'num_variables': len(variable_names),
            'variables': variable_names,
            'time': elapsed,
            'records': len(result_gdf),
            'values_per_sec': len(result_gdf) * len(variable_names) / elapsed
        }
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("\n" + "=" * 70)
    print("MULTI-VARIABLE QUERY PERFORMANCE BENCHMARK")
    print("=" * 70)
    print()
    
    # Load test data
    print("Loading test data...")
    gdf = load_spatiotemporal_csv('../notebooks/spatio_temporal.csv')
    
    # Use small subset for quick testing
    test_size = 5
    gdf_test = gdf.head(test_size).copy()
    
    print(f"Dataset: {len(gdf_test)} records")
    print(f"Date range: {gdf_test['time_UTC'].min()} to {gdf_test['time_UTC'].max()}")
    unique_coords = gdf_test.geometry.apply(lambda g: (round(g.x, 6), round(g.y, 6))).nunique()
    print(f"Unique coordinates: {unique_coords}")
    print()
    
    # Test configurations
    variable_sets = [
        ["Ta_K"],                    # 1 variable
        ["Ta_K", "SM"],              # 2 variables
        ["Ta_K", "SM", "LAI"],       # 3 variables
    ]
    
    results = []
    
    # Run benchmark for each variable set
    for var_set in variable_sets:
        print("=" * 70)
        print(f"TEST: {len(var_set)} VARIABLE{'S' if len(var_set) > 1 else ''}")
        print("=" * 70)
        
        result = benchmark_multi_variable(gdf_test, var_set)
        if result:
            results.append(result)
        
        print()
    
    # Summary
    print("=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print()
    
    if results:
        df_results = pd.DataFrame(results)
        print(df_results[['num_variables', 'time', 'values_per_sec']].to_string(index=False))
        print()
        
        # Analyze speedup
        if len(results) >= 2:
            baseline = results[0]  # Single variable
            
            print("MULTI-VARIABLE EFFICIENCY:")
            print(f"  Baseline (1 variable):  {baseline['time']:.1f}s")
            print()
            
            for result in results[1:]:
                n_vars = result['num_variables']
                actual_time = result['time']
                expected_time = baseline['time'] * n_vars
                speedup = expected_time / actual_time
                time_saved = expected_time - actual_time
                
                print(f"  {n_vars} variables ({', '.join(result['variables'])}):")
                print(f"    Expected (sequential): {expected_time:.1f}s ({n_vars} × {baseline['time']:.1f}s)")
                print(f"    Actual (parallel):     {actual_time:.1f}s")
                print(f"    🚀 Speedup:            {speedup:.2f}x")
                print(f"    ⏱️  Time saved:         {time_saved:.1f}s ({time_saved/expected_time*100:.0f}% reduction)")
                print()
            
            # Extrapolation to full dataset
            print("EXTRAPOLATION TO FULL DATASET (1,065 records):")
            print("-" * 70)
            scale_factor = 1065 / len(gdf_test)
            
            for result in results:
                n_vars = result['num_variables']
                time_small = result['time']
                time_full = time_small * scale_factor
                
                print(f"  {n_vars} variable{'s' if n_vars > 1 else ''}: "
                      f"{time_full:.0f}s ({time_full/60:.1f} min)")
            
            print()
            
            # Show impact of optimization
            if len(results) == 3:
                sequential_3vars = results[0]['time'] * 3 * scale_factor
                optimized_3vars = results[2]['time'] * scale_factor
                
                print("IMPACT FOR TYPICAL USE CASE (3 variables, 1,065 records):")
                print(f"  Sequential processing: {sequential_3vars:.0f}s ({sequential_3vars/60:.1f} min)")
                print(f"  Multi-variable query:  {optimized_3vars:.0f}s ({optimized_3vars/60:.1f} min)")
                print(f"  Time saved:           {sequential_3vars - optimized_3vars:.0f}s "
                      f"({(sequential_3vars - optimized_3vars)/60:.1f} min)")
                print(f"  Speedup:              {sequential_3vars / optimized_3vars:.2f}x")
    
    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
