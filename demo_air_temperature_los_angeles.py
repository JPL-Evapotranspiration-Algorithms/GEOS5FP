"""
GEOS-5 FP Air Temperature Retrieval Demo for Los Angeles
========================================================

This demonstration shows how to retrieve air temperature values from GEOS-5 FP 
at a single spatial point location in Los Angeles for a set of datetimes.

The demo covers:
1. Setting up coordinate points in Los Angeles
2. Retrieving air temperature data for multiple time points
3. Displaying the results as a time series
4. Showing both Kelvin and Celsius values

Location: Downtown Los Angeles (approximate coordinates)
Latitude: 34.0522° N
Longitude: -118.2437° W
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from GEOS5FP import GEOS5FP
import numpy as np

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Demonstrate retrieving air temperature values from GEOS-5 FP at a single 
    spatial point location in Los Angeles for multiple datetimes.
    """
    
    print("GEOS-5 FP Air Temperature Retrieval Demo for Los Angeles")
    print("=" * 60)
    
    # Initialize GEOS-5 FP connection
    print("\n1. Initializing GEOS-5 FP connection...")
    geos5fp = GEOS5FP()
    print(f"   Connected to: {geos5fp.remote}")
    print(f"   Download directory: {geos5fp.download_directory}")
    
    # Define Los Angeles coordinates
    # Downtown LA: 34.0522° N, 118.2437° W
    los_angeles_lat = 34.0522
    los_angeles_lon = -118.2437
    
    print(f"\n2. Target location: Los Angeles, CA")
    print(f"   Latitude: {los_angeles_lat}° N")
    print(f"   Longitude: {los_angeles_lon}° W")
    
    # Create a list of coordinate tuples for the single point
    # The GEOS5FP library expects (longitude, latitude) order
    points = [(los_angeles_lon, los_angeles_lat)]
    
    print(f"\n3. Point geometry: ({los_angeles_lon}, {los_angeles_lat})")
    
    # Define a set of datetime points to retrieve data for
    # Let's get data for several times over a few days
    base_date = datetime(2024, 8, 15, 12, 0)  # August 15, 2024 at noon UTC
    
    # Generate time series: every 6 hours for 3 days
    time_points = []
    for i in range(13):  # 13 points = 3 days of 6-hourly data + 1
        time_points.append(base_date + timedelta(hours=i * 6))
    
    print(f"\n4. Time series configuration:")
    print(f"   Start time: {time_points[0]} UTC")
    print(f"   End time: {time_points[-1]} UTC")
    print(f"   Number of time points: {len(time_points)}")
    print(f"   Interval: 6 hours")
    
    print("\n5. Retrieving air temperature data...")
    print("   This may take several minutes as data is downloaded and processed...")
    
    try:
        # Method 1: Using the consolidated get_variable method for time series
        print("\n   Using consolidated get_variable method for T2M...")
        
        # Retrieve air temperature in Kelvin using the new consolidated method
        temperature_data = geos5fp.get_variable(
            variable_name='T2M',
            time_UTC=time_points,
            geometry=points
        )
        
        print(f"\n6. Results Summary:")
        print(f"   Data shape: {temperature_data.shape}")
        print(f"   Time range: {temperature_data.index.min()} to {temperature_data.index.max()}")
        print(f"   Data type: {type(temperature_data)}")
        
        # Convert temperature from Kelvin to Celsius
        temperature_celsius = temperature_data.iloc[:, 0] - 273.15
        
        # Create a comprehensive results DataFrame
        results_df = pd.DataFrame({
            'datetime_UTC': temperature_data.index,
            'temperature_K': temperature_data.iloc[:, 0],
            'temperature_C': temperature_celsius,
            'temperature_F': (temperature_celsius * 9/5) + 32
        })
        
        print(f"\n7. Air Temperature Time Series for Los Angeles:")
        print("   " + "=" * 70)
        print("   {:^20} {:>12} {:>12} {:>12}".format(
            "DateTime (UTC)", "Temp (K)", "Temp (°C)", "Temp (°F)"
        ))
        print("   " + "-" * 70)
        
        for _, row in results_df.iterrows():
            print("   {:^20} {:>12.2f} {:>12.2f} {:>12.2f}".format(
                row['datetime_UTC'].strftime('%Y-%m-%d %H:%M'),
                row['temperature_K'],
                row['temperature_C'], 
                row['temperature_F']
            ))
        
        # Calculate some basic statistics
        print(f"\n8. Temperature Statistics:")
        print(f"   Minimum: {results_df['temperature_C'].min():.2f} °C ({results_df['temperature_F'].min():.2f} °F)")
        print(f"   Maximum: {results_df['temperature_C'].max():.2f} °C ({results_df['temperature_F'].max():.2f} °F)")
        print(f"   Average: {results_df['temperature_C'].mean():.2f} °C ({results_df['temperature_F'].mean():.2f} °F)")
        print(f"   Standard deviation: {results_df['temperature_C'].std():.2f} °C")
        
        # Method 2: Alternative approach using individual time points
        print(f"\n9. Alternative Method - Single Time Point Retrieval:")
        print("   Demonstrating retrieval of a single time point...")
        
        # Get temperature for just the first time point
        single_time = time_points[0]
        single_temp = geos5fp.get_variable(
            variable_name='T2M',
            time_UTC=single_time,
            geometry=points
        )
        
        single_temp_celsius = single_temp.iloc[0] - 273.15
        
        print(f"   Single time point: {single_time} UTC")
        print(f"   Temperature: {single_temp_celsius:.2f} °C ({(single_temp_celsius * 9/5) + 32:.2f} °F)")
        
        # Method 3: Using the legacy method (Ta_K) for comparison
        print(f"\n10. Legacy Method Comparison:")
        print("    Using Ta_K method for single point...")
        
        try:
            # Note: Ta_K expects RasterGeometry, so we'll use the single time method
            legacy_temp_K = geos5fp.Ta_K(time_UTC=single_time)
            # Sample the raster at our point location
            legacy_temp_value = legacy_temp_K.sample([(los_angeles_lon, los_angeles_lat)])[0]
            legacy_temp_celsius = legacy_temp_value - 273.15
            
            print(f"    Legacy method result: {legacy_temp_celsius:.2f} °C")
            print(f"    Difference from new method: {abs(legacy_temp_celsius - single_temp_celsius):.4f} °C")
            
        except Exception as e:
            print(f"    Legacy method note: {e}")
            print("    (This is expected as the legacy method uses different geometry handling)")
        
        print(f"\n11. Data Source Information:")
        if hasattr(temperature_data, 'filenames') or 'filenames' in str(temperature_data):
            print("    Data retrieved from GEOS-5 FP granule files")
        print("    Product: tavg1_2d_slv_Nx (1-hourly time-averaged)")
        print("    Variable: T2M (2-meter air temperature)")
        print("    Spatial resolution: ~25 km (0.25° x 0.3125°)")
        print("    Temporal resolution: 1-hour")
        
        print(f"\n12. Success!")
        print("    Air temperature time series successfully retrieved from GEOS-5 FP")
        print("    The demonstration shows how to extract point-based meteorological")
        print("    data for specific locations and time periods.")
        
        return results_df
        
    except Exception as e:
        print(f"\nError during data retrieval: {e}")
        logger.exception("Detailed error information:")
        
        # Provide troubleshooting guidance
        print(f"\nTroubleshooting:")
        print(f"  1. Check internet connection")
        print(f"  2. Verify that the requested dates have available data")
        print(f"  3. The GEOS-5 FP archive may have delays for recent dates")
        print(f"  4. Try with older dates (e.g., 2023 data)")
        
        # Try with a more recent but likely available date
        print(f"\n  Suggestion: Try with an older date like 2023-08-15")
        
        return None


if __name__ == "__main__":
    # Execute the demonstration
    results = main()
    
    if results is not None:
        print(f"\nDemo completed successfully!")
        print(f"Results DataFrame shape: {results.shape}")
        print(f"You can now use the 'results' DataFrame for further analysis.")
    else:
        print(f"\nDemo completed with errors. See troubleshooting suggestions above.")