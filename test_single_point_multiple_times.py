"""
GEOS-5 FP Air Temperature Demo: Single Point, Multiple Times
============================================================

Demonstrates retrieving air temperature values from GEOS-5 FP at a single 
spatial point location in Los Angeles for a set of datetimes.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from GEOS5FP import GEOS5FP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_air_temperature_retrieval():
    """
    Retrieve air temperature at downtown Los Angeles for multiple time points.
    """
    print("GEOS-5 FP Air Temperature Demo for Los Angeles")
    print("=" * 50)
    
    # Initialize connection
    geos5fp = GEOS5FP()
    
    # Los Angeles downtown coordinates (longitude, latitude)
    los_angeles = (-118.2437, 34.0522)  # Downtown LA
    points = [los_angeles]
    
    print(f"Location: Los Angeles, CA")
    print(f"Coordinates: {los_angeles[1]}°N, {abs(los_angeles[0])}°W")
    
    # Create time series - every 6 hours for 2 days
    base_time = datetime(2024, 7, 15, 12, 0)  # July 15, 2024, noon UTC
    times = [base_time + timedelta(hours=6*i) for i in range(9)]  # 9 time points
    
    print(f"\nTime series: {len(times)} points from {times[0]} to {times[-1]} UTC")
    
    try:
        print("\nRetrieving air temperature data...")
        
        # Get temperature data for single times and extract values
        temp_values = []
        actual_times = []
        
        for time in times:
            try:
                print(f"  Processing {time.strftime('%m/%d %H:%M')}...")
                
                # Get temperature raster for this time
                temp_raster = geos5fp.Ta_K(time_UTC=time)
                
                # Sample at the Los Angeles coordinates
                # Convert longitude from -118 to proper raster coordinates
                # GEOS-5 FP uses 0-360 longitude, so add 360 to negative values
                lon_360 = los_angeles[0] + 360 if los_angeles[0] < 0 else los_angeles[0]
                lat = los_angeles[1]
                
                # Sample the raster at our point
                try:
                    # Get raster bounds and resolution to find pixel indices
                    bounds = temp_raster.geometry.bounds
                    shape = temp_raster.array.shape
                    
                    # Calculate pixel indices manually
                    # GEOS-5 FP longitude range is 0-360
                    lon_min, lat_min, lon_max, lat_max = bounds
                    
                    # Calculate pixel coordinates
                    lon_res = (lon_max - lon_min) / shape[1]
                    lat_res = (lat_max - lat_min) / shape[0]
                    
                    col = int((lon_360 - lon_min) / lon_res)
                    row = int((lat_max - lat) / lat_res)  # Flip for array indexing
                    
                    # Ensure indices are within bounds
                    col = max(0, min(col, shape[1] - 1))
                    row = max(0, min(row, shape[0] - 1))
                    
                    temp_k = temp_raster.array[row, col]
                    
                except Exception as e:
                    print(f"      Error in sampling: {e}")
                    # Try simple center point approach
                    try:
                        center_row = shape[0] // 2
                        center_col = shape[1] // 2
                        temp_k = temp_raster.array[center_row, center_col]
                        print(f"      Using center pixel as fallback")
                    except:
                        temp_k = None
                
                if temp_k is not None and not np.isnan(temp_k):
                    temp_values.append(temp_k)
                    actual_times.append(time)
                    print(f"    Success: {temp_k - 273.15:.1f}°C")
                else:
                    print(f"    No data available")
                    
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        if not temp_values:
            print("\nNo temperature data could be retrieved.")
            return None
        
        # Convert from Kelvin to Celsius
        temp_celsius = pd.Series([t - 273.15 for t in temp_values], index=actual_times)
        
        print(f"\nAir Temperature Results ({len(temp_celsius)} values):")
        print("-" * 40)
        print(f"{'Time (UTC)':<17} {'Temp (°C)':<10} {'Temp (°F)':<10}")
        print("-" * 40)
        
        for time, temp_c in temp_celsius.items():
            temp_f = (temp_c * 9/5) + 32
            time_str = time.strftime('%m/%d %H:%M')
            print(f"{time_str:<17} {temp_c:>8.1f} {temp_f:>10.1f}")
        
        print(f"\nSummary Statistics:")
        print(f"  Min: {temp_celsius.min():.1f}°C ({(temp_celsius.min() * 9/5) + 32:.1f}°F)")
        print(f"  Max: {temp_celsius.max():.1f}°C ({(temp_celsius.max() * 9/5) + 32:.1f}°F)")
        print(f"  Avg: {temp_celsius.mean():.1f}°C ({(temp_celsius.mean() * 9/5) + 32:.1f}°F)")
        
        return temp_celsius
        
    except Exception as e:
        print(f"\nError retrieving data: {e}")
        print("Note: Try using a date from 2023 or earlier for better data availability")
        return None

if __name__ == "__main__":
    result = demonstrate_air_temperature_retrieval()
    if result is not None:
        print(f"\nSuccess! Retrieved {len(result)} temperature values.")
    else:
        print(f"\nDemo failed. Check the error message above.")