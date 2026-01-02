"""
Test script to verify that single-variable non-raster queries return numpy arrays.
"""
from datetime import datetime
import numpy as np
from shapely.geometry import Point, MultiPoint
from GEOS5FP import GEOS5FP

def test_single_point_query():
    """Test that a single point query returns a numpy array."""
    print("Testing single point query...")
    
    conn = GEOS5FP()
    time_UTC = datetime(2024, 11, 15, 12, 0, 0)
    geometry = Point(-118.25, 34.05)  # Los Angeles
    
    result = conn.COT(time_UTC=time_UTC, geometry=geometry)
    
    print(f"Type of result: {type(result)}")
    print(f"Result: {result}")
    
    assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result)}"
    print("✓ Single point query returns numpy array")
    
    return result

def test_multipoint_query():
    """Test that a MultiPoint query returns a numpy array."""
    print("\nTesting MultiPoint query...")
    
    conn = GEOS5FP()
    time_UTC = datetime(2024, 11, 15, 12, 0, 0)
    
    # Create MultiPoint with multiple locations
    points = [
        Point(-118.25, 34.05),  # Los Angeles
        Point(-122.42, 37.77),  # San Francisco
        Point(-73.99, 40.75)    # New York
    ]
    geometry = MultiPoint(points)
    
    result = conn.AOT(time_UTC=time_UTC, geometry=geometry)
    
    print(f"Type of result: {type(result)}")
    print(f"Result shape: {result.shape if isinstance(result, np.ndarray) else 'N/A'}")
    print(f"Result: {result}")
    
    assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result)}"
    assert len(result) == len(points), f"Expected {len(points)} values, got {len(result)}"
    print("✓ MultiPoint query returns numpy array with correct length")
    
    return result

def test_vapor_query():
    """Test vapor_gccm query returns numpy array."""
    print("\nTesting vapor_gccm query...")
    
    conn = GEOS5FP()
    time_UTC = datetime(2024, 11, 15, 12, 0, 0)
    geometry = Point(-118.25, 34.05)
    
    result = conn.vapor_gccm(time_UTC=time_UTC, geometry=geometry)
    
    print(f"Type of result: {type(result)}")
    print(f"Result: {result}")
    
    assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result)}"
    print("✓ vapor_gccm query returns numpy array")
    
    return result

def test_ozone_query():
    """Test ozone_cm query returns numpy array."""
    print("\nTesting ozone_cm query...")
    
    conn = GEOS5FP()
    time_UTC = datetime(2024, 11, 15, 12, 0, 0)
    geometry = Point(-118.25, 34.05)
    
    result = conn.ozone_cm(time_UTC=time_UTC, geometry=geometry)
    
    print(f"Type of result: {type(result)}")
    print(f"Result: {result}")
    
    assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result)}"
    print("✓ ozone_cm query returns numpy array")
    
    return result

if __name__ == "__main__":
    print("=" * 60)
    print("Testing GEOS5FP numpy array return types")
    print("=" * 60)
    
    try:
        test_single_point_query()
        test_multipoint_query()
        test_vapor_query()
        test_ozone_query()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
