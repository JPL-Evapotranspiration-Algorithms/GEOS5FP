#!/usr/bin/env python3
"""
Test script to verify that subprocess isolation works correctly.
"""

import os
import tempfile
import numpy as np
from GEOS5FP.validate_GEOS5FP_NetCDF_file import (
    validate_GEOS5FP_NetCDF_file,
    safe_validate_GEOS5FP_NetCDF_file,
    GEOS5FPValidationResult
)

def create_dummy_netcdf(filename: str):
    """Create a simple dummy NetCDF file for testing."""
    try:
        import xarray as xr
        
        # Create simple test data
        data = xr.Dataset({
            'temperature': (['time', 'lat', 'lon'], np.random.rand(1, 10, 10)),
            'pressure': (['time', 'lat', 'lon'], np.random.rand(1, 10, 10))
        }, coords={
            'time': [0],
            'lat': np.linspace(-90, 90, 10),
            'lon': np.linspace(-180, 180, 10)
        })
        
        data.to_netcdf(filename)
        return True
    except ImportError:
        print("xarray not available, cannot create test NetCDF file")
        return False

def test_subprocess_validation():
    """Test subprocess validation functionality."""
    print("Testing subprocess validation...")
    
    # Create a temporary NetCDF file
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Create test file
        if not create_dummy_netcdf(tmp_path):
            print("Skipping test - cannot create test file")
            return
        
        print(f"Created test file: {tmp_path}")
        
        # Test direct validation
        print("\n1. Testing direct validation...")
        result_direct = validate_GEOS5FP_NetCDF_file(tmp_path, use_subprocess=False)
        print(f"Direct validation result: {result_direct.is_valid}")
        print(f"Direct validation issues: {len(result_direct.issues)}")
        
        # Test subprocess validation
        print("\n2. Testing subprocess validation...")
        result_subprocess = validate_GEOS5FP_NetCDF_file(tmp_path, use_subprocess=True)
        print(f"Subprocess validation result: {result_subprocess.is_valid}")
        print(f"Subprocess validation issues: {len(result_subprocess.issues)}")
        
        # Test safe validation convenience function
        print("\n3. Testing safe validation convenience function...")
        result_safe = safe_validate_GEOS5FP_NetCDF_file(tmp_path)
        print(f"Safe validation result: {result_safe.is_valid}")
        print(f"Safe validation issues: {len(result_safe.issues)}")
        
        # Compare results
        print("\n4. Comparing results...")
        print(f"Direct == Subprocess: {result_direct.is_valid == result_subprocess.is_valid}")
        print(f"Direct == Safe: {result_direct.is_valid == result_safe.is_valid}")
        
        # Test with non-existent file
        print("\n5. Testing with non-existent file...")
        result_missing = validate_GEOS5FP_NetCDF_file("nonexistent.nc", use_subprocess=True)
        print(f"Missing file validation: {result_missing.is_valid}")
        print(f"Missing file error: {result_missing.error}")
        
        print("\n✅ All tests completed successfully!")
        
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print(f"Cleaned up test file: {tmp_path}")

def test_timeout_handling():
    """Test timeout handling in subprocess validation."""
    print("\nTesting timeout handling...")
    
    # Test with very short timeout to trigger timeout
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        if not create_dummy_netcdf(tmp_path):
            print("Skipping timeout test - cannot create test file")
            return
        
        print(f"Testing with very short timeout (0.001 seconds)...")
        result = validate_GEOS5FP_NetCDF_file(
            tmp_path, 
            use_subprocess=True, 
            timeout_seconds=0.001
        )
        
        print(f"Timeout test result: {result.is_valid}")
        print(f"Timeout test error: {result.error}")
        
        if "timeout" in str(result.error).lower():
            print("✅ Timeout handling works correctly!")
        else:
            print("⚠️  Timeout might not have been triggered (file processed too quickly)")
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    print("GEOS-5 FP Subprocess Validation Test")
    print("=" * 50)
    
    test_subprocess_validation()
    test_timeout_handling()
    
    print("\n" + "=" * 50)
    print("Test completed!")