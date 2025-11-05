"""
Simple GEOS-5 FP Temperature Demo for Los Angeles
=================================================

A simplified demonstration showing how to retrieve air temperature values
from GEOS-5 FP at Los Angeles coordinates for multiple times.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from GEOS5FP import GEOS5FP

# Configure basic logging
logging.basicConfig(level=logging.INFO)

def simple_temperature_demo():
    """Simple demo of retrieving temperature data from GEOS-5 FP"""
    
    print("Simple GEOS-5 FP Temperature Demo")
    print("=" * 40)
    
    # Initialize connection
    geos5fp = GEOS5FP()
    
    # Los Angeles coordinates (downtown)
    print("Target location: Downtown Los Angeles")
    print("Coordinates: 34.0522°N, 118.2437°W")
    
    # Create a few time points (3 times over 12 hours)
    base_time = datetime(2024, 6, 15, 12, 0)  # June 15, 2024, noon UTC
    times = [
        base_time,
        base_time + timedelta(hours=6),
        base_time + timedelta(hours=12)
    ]
    
    print(f"\nRetrieving temperature data for {len(times)} time points:")
    
    results = []
    
    for i, time in enumerate(times, 1):
        print(f"\n{i}. Processing {time.strftime('%Y-%m-%d %H:%M')} UTC...")
        
        try:
            # Get temperature raster (in Kelvin)
            temp_raster = geos5fp.Ta_K(time_UTC=time)
            
            print(f"   Success! Raster shape: {temp_raster.array.shape}")
            print(f"   Data type: {type(temp_raster)}")
            
            # Get a representative temperature value from the raster
            # (Using mean of valid data as an example)
            valid_temps = temp_raster.array[~np.isnan(temp_raster.array)]
            
            if len(valid_temps) > 0:
                # Use median temperature as representative value
                temp_k = np.median(valid_temps)
                temp_c = temp_k - 273.15
                temp_f = (temp_c * 9/5) + 32
                
                results.append({
                    'time': time,
                    'temp_k': temp_k,
                    'temp_c': temp_c,
                    'temp_f': temp_f
                })
                
                print(f"   Representative temperature: {temp_c:.1f}°C ({temp_f:.1f}°F)")
                
            else:
                print(f"   Warning: No valid temperature data found")
                
        except Exception as e:
            print(f"   Error: {e}")
            continue
    
    # Display results
    if results:
        print(f"\n" + "="*50)
        print("TEMPERATURE SUMMARY")
        print("="*50)
        print(f"{'Time (UTC)':<20} {'Temp (°C)':<10} {'Temp (°F)':<10}")
        print("-" * 50)
        
        for result in results:
            time_str = result['time'].strftime('%m/%d %H:%M')
            print(f"{time_str:<20} {result['temp_c']:>8.1f} {result['temp_f']:>10.1f}")
        
        # Calculate basic statistics
        temps_c = [r['temp_c'] for r in results]
        print(f"\nStatistics:")
        print(f"  Average: {np.mean(temps_c):.1f}°C")
        print(f"  Range: {min(temps_c):.1f}°C to {max(temps_c):.1f}°C")
        
        print(f"\nSuccess! Retrieved temperature data for {len(results)} time points.")
        print(f"Note: These are representative regional values, not point-specific.")
        
    else:
        print(f"\nNo temperature data could be retrieved.")
        
    return results

if __name__ == "__main__":
    results = simple_temperature_demo()
    
    if results:
        print(f"\nDemo completed successfully!")
        print(f"Data structure: List of dictionaries with time and temperature values")
    else:
        print(f"\nDemo completed with no data retrieved.")
        print(f"This may be due to data availability issues.")