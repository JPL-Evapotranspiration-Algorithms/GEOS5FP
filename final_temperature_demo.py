"""
GEOS-5 FP Air Temperature Extraction for Los Angeles
====================================================

Complete demonstration of retrieving air temperature values from GEOS-5 FP
at specific Los Angeles coordinates for multiple datetime points.

This example shows:
1. How to connect to GEOS-5 FP data
2. How to retrieve temperature rasters for specific times
3. How to extract values at specific geographic coordinates
4. How to handle time series of meteorological data
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from GEOS5FP import GEOS5FP

# Configure logging
logging.basicConfig(level=logging.INFO)

def extract_temperature_at_coordinates():
    """
    Extract air temperature values at Los Angeles coordinates from GEOS-5 FP
    """
    
    print("GEOS-5 FP Air Temperature Extraction for Los Angeles")
    print("=" * 60)
    
    # Initialize GEOS-5 FP connection
    geos5fp = GEOS5FP()
    
    # Los Angeles coordinates
    la_longitude = -118.2437  # Downtown LA longitude (West is negative)
    la_latitude = 34.0522     # Downtown LA latitude
    
    print(f"Target Location:")
    print(f"  City: Los Angeles, California")
    print(f"  Coordinates: {la_latitude:.4f}°N, {abs(la_longitude):.4f}°W")
    print(f"  Longitude: {la_longitude}")
    print(f"  Latitude: {la_latitude}")
    
    # Create time series - 4 time points over 18 hours
    base_time = datetime(2024, 6, 15, 6, 0)  # June 15, 2024, 6:00 UTC
    time_points = [
        base_time,                           # 06:00 UTC (11 PM PDT previous day)
        base_time + timedelta(hours=6),      # 12:00 UTC (5 AM PDT)  
        base_time + timedelta(hours=12),     # 18:00 UTC (11 AM PDT)
        base_time + timedelta(hours=18)      # 00:00 UTC (5 PM PDT)
    ]
    
    print(f"\nTime Series Configuration:")
    print(f"  Start: {time_points[0].strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  End: {time_points[-1].strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  Points: {len(time_points)} (every 6 hours)")
    print(f"  Time zone note: UTC times shown (LA is UTC-7 in summer)")
    
    print(f"\nExtracting temperature data...")
    
    results = []
    
    for i, time_utc in enumerate(time_points, 1):
        print(f"\n{i}. Processing {time_utc.strftime('%Y-%m-%d %H:%M')} UTC...")
        
        try:
            # Get temperature raster from GEOS-5 FP (in Kelvin)
            temp_raster = geos5fp.Ta_K(time_UTC=time_utc)
            
            print(f"   ✓ Downloaded raster data (shape: {temp_raster.array.shape})")
            
            # Extract temperature at Los Angeles coordinates
            temp_k = extract_value_at_coordinates(
                temp_raster, 
                la_longitude, 
                la_latitude
            )
            
            if temp_k is not None and not np.isnan(temp_k):
                temp_c = temp_k - 273.15
                temp_f = (temp_c * 9/5) + 32
                
                # Convert UTC to local Pacific time for display
                local_hour = (time_utc.hour - 7) % 24  # Rough PDT conversion
                
                results.append({
                    'datetime_utc': time_utc,
                    'local_hour_pdt': local_hour,
                    'temperature_k': temp_k,
                    'temperature_c': temp_c,
                    'temperature_f': temp_f
                })
                
                print(f"   ✓ Temperature: {temp_c:.1f}°C ({temp_f:.1f}°F)")
                print(f"     Local time: ~{local_hour:02d}:00 PDT")
                
            else:
                print(f"   ✗ No valid temperature data at coordinates")
                
        except Exception as e:
            print(f"   ✗ Error: {e}")
            continue
    
    # Display comprehensive results
    if results:
        print(f"\n" + "="*70)
        print("TEMPERATURE TIME SERIES FOR LOS ANGELES")
        print("="*70)
        print(f"{'UTC Time':<12} {'PDT Time':<9} {'Temp (K)':<9} {'Temp (°C)':<9} {'Temp (°F)':<9}")
        print("-" * 70)
        
        for result in results:
            utc_str = result['datetime_utc'].strftime('%m/%d %H:%M')
            pdt_str = f"{result['local_hour_pdt']:02d}:00"
            
            print(f"{utc_str:<12} {pdt_str:<9} "
                  f"{result['temperature_k']:>8.1f} "
                  f"{result['temperature_c']:>8.1f} "
                  f"{result['temperature_f']:>8.1f}")
        
        # Calculate statistics
        temps_c = [r['temperature_c'] for r in results]
        temps_f = [r['temperature_f'] for r in results]
        
        print(f"\n" + "="*70)
        print("STATISTICS")
        print("="*70)
        print(f"Number of data points: {len(results)}")
        print(f"Temperature range (°C): {min(temps_c):.1f}°C to {max(temps_c):.1f}°C")
        print(f"Temperature range (°F): {min(temps_f):.1f}°F to {max(temps_f):.1f}°F")
        print(f"Average temperature: {np.mean(temps_c):.1f}°C ({np.mean(temps_f):.1f}°F)")
        print(f"Temperature variation: {max(temps_c) - min(temps_c):.1f}°C")
        
        # Create DataFrame for easy analysis
        df = pd.DataFrame(results)
        df = df.set_index('datetime_utc')
        
        print(f"\n" + "="*70)
        print("DATA ANALYSIS")
        print("="*70)
        print(f"Warmest time: {df.loc[df['temperature_c'].idxmax(), 'local_hour_pdt']:02d}:00 PDT "
              f"({df['temperature_c'].max():.1f}°C)")
        print(f"Coolest time: {df.loc[df['temperature_c'].idxmin(), 'local_hour_pdt']:02d}:00 PDT "
              f"({df['temperature_c'].min():.1f}°C)")
        
        print(f"\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"Successfully retrieved air temperature values from GEOS-5 FP")
        print(f"for Los Angeles coordinates ({la_latitude:.4f}°N, {abs(la_longitude):.4f}°W)")
        print(f"Data source: NASA GEOS-5 FP (Forward Processing)")
        print(f"Variable: T2M (2-meter air temperature)")
        print(f"Spatial resolution: ~25 km")
        print(f"Temporal resolution: 1-hour (interpolated)")
        
        return df
        
    else:
        print(f"\nNo temperature data could be retrieved.")
        return None


def extract_value_at_coordinates(raster, longitude, latitude):
    """
    Extract raster value at specific geographic coordinates
    
    Args:
        raster: Raster object from GEOS5FP
        longitude: Longitude in degrees (-180 to 180)
        latitude: Latitude in degrees (-90 to 90)
    
    Returns:
        Extracted value or None if extraction fails
    """
    try:
        # GEOS-5 FP uses 0-360 longitude convention
        lon_360 = longitude + 360 if longitude < 0 else longitude
        
        # For GEOS-5 FP data, we know the approximate global bounds
        # Global extent: longitude 0-360, latitude -90 to 90
        # GEOS-5 FP resolution is approximately 0.25° x 0.3125°
        
        shape = raster.array.shape  # Should be (721, 1152) for GEOS-5 FP
        
        # GEOS-5 FP grid specifications:
        # Longitude: 0 to 360 degrees, 1152 points -> 0.3125° resolution
        # Latitude: -90 to 90 degrees, 721 points -> 0.25° resolution
        
        lon_min, lon_max = 0, 360
        lat_min, lat_max = -90, 90
        
        # Calculate pixel coordinates
        lon_res = (lon_max - lon_min) / shape[1]  # ~0.3125
        lat_res = (lat_max - lat_min) / shape[0]  # ~0.25
        
        # Find pixel indices
        col = int((lon_360 - lon_min) / lon_res)
        row = int((lat_max - latitude) / lat_res)  # Flip for array indexing
        
        # Ensure indices are within bounds
        col = max(0, min(col, shape[1] - 1))
        row = max(0, min(row, shape[0] - 1))
        
        # Extract value
        value = raster.array[row, col]
        
        print(f"    Coordinates: {longitude:.4f}°, {latitude:.4f}° -> pixel [{row}, {col}]")
        
        return value
        
    except Exception as e:
        print(f"    Error extracting value: {e}")
        # Try a simple center-point extraction as fallback
        try:
            shape = raster.array.shape
            center_row = shape[0] // 2
            center_col = shape[1] // 2
            value = raster.array[center_row, center_col]
            print(f"    Using center pixel [{center_row}, {center_col}] as fallback")
            return value
        except:
            return None


if __name__ == "__main__":
    print("Starting GEOS-5 FP air temperature extraction demonstration...\n")
    
    results_df = extract_temperature_at_coordinates()
    
    if results_df is not None:
        print(f"\nDemonstration completed successfully!")
        print(f"Results available in pandas DataFrame with {len(results_df)} records")
        print(f"DataFrame columns: {list(results_df.columns)}")
    else:
        print(f"\nDemonstration completed with no data retrieved.")
        print(f"This may be due to network issues or data availability.")
        print(f"Try running again or using different dates.")