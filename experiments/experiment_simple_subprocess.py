#!/usr/bin/env python3
"""
Simple test script to verify subprocess isolation without external dependencies.
"""

import os
import tempfile
from GEOS5FP.validate_GEOS5FP_NetCDF_file import (
    validate_GEOS5FP_NetCDF_file,
    safe_validate_GEOS5FP_NetCDF_file
)

def test_basic_functionality():
    """Test basic functionality with non-existent files."""
    print("Testing basic subprocess functionality...")
    
    # Test with non-existent file (should fail gracefully)
    nonexistent_file = "definitely_does_not_exist.nc"
    
    print("\n1. Testing direct validation with non-existent file...")
    result_direct = validate_GEOS5FP_NetCDF_file(nonexistent_file, use_subprocess=False)
    print(f"Direct result - Valid: {result_direct.is_valid}")
    print(f"Direct result - Error: {result_direct.error}")
    
    print("\n2. Testing subprocess validation with non-existent file...")
    result_subprocess = validate_GEOS5FP_NetCDF_file(nonexistent_file, use_subprocess=True)
    print(f"Subprocess result - Valid: {result_subprocess.is_valid}")
    print(f"Subprocess result - Error: {result_subprocess.error}")
    
    print("\n3. Testing safe validation with non-existent file...")
    result_safe = safe_validate_GEOS5FP_NetCDF_file(nonexistent_file)
    print(f"Safe result - Valid: {result_safe.is_valid}")
    print(f"Safe result - Error: {result_safe.error}")
    
    # All should be invalid due to file not existing
    assert not result_direct.is_valid, "Direct validation should fail for non-existent file"
    assert not result_subprocess.is_valid, "Subprocess validation should fail for non-existent file"
    assert not result_safe.is_valid, "Safe validation should fail for non-existent file"
    
    print("\n✅ All methods correctly identified non-existent file as invalid!")

def test_empty_file():
    """Test with an empty file."""
    print("\n" + "="*50)
    print("Testing with empty file...")
    
    # Create an empty file
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        tmp_path = tmp.name
        # File is created but empty
    
    try:
        print(f"Created empty file: {tmp_path}")
        
        print("\n1. Testing direct validation with empty file...")
        result_direct = validate_GEOS5FP_NetCDF_file(tmp_path, use_subprocess=False)
        print(f"Direct result - Valid: {result_direct.is_valid}")
        print(f"Direct result - Error: {result_direct.error}")
        
        print("\n2. Testing subprocess validation with empty file...")
        result_subprocess = validate_GEOS5FP_NetCDF_file(tmp_path, use_subprocess=True)
        print(f"Subprocess result - Valid: {result_subprocess.is_valid}")
        print(f"Subprocess result - Error: {result_subprocess.error}")
        
        # Both should be invalid (empty file)
        assert not result_direct.is_valid, "Direct validation should fail for empty file"
        assert not result_subprocess.is_valid, "Subprocess validation should fail for empty file"
        
        print("\n✅ Both methods correctly identified empty file as invalid!")
            
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print(f"Cleaned up empty file: {tmp_path}")

def test_text_file():
    """Test with a text file (not NetCDF)."""
    print("\n" + "="*50)
    print("Testing with text file...")
    
    # Create a text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.nc', delete=False) as tmp:
        tmp.write("This is not a NetCDF file\nJust some text content\n")
        tmp_path = tmp.name
    
    try:
        print(f"Created text file: {tmp_path}")
        
        print("\n1. Testing direct validation with text file...")
        result_direct = validate_GEOS5FP_NetCDF_file(tmp_path, use_subprocess=False)
        print(f"Direct result - Valid: {result_direct.is_valid}")
        print(f"Direct result - Error: {result_direct.error}")
        
        print("\n2. Testing subprocess validation with text file...")
        result_subprocess = validate_GEOS5FP_NetCDF_file(tmp_path, use_subprocess=True)
        print(f"Subprocess result - Valid: {result_subprocess.is_valid}")
        print(f"Subprocess result - Error: {result_subprocess.error}")
        
        # Both should be invalid (not a NetCDF file)
        assert not result_direct.is_valid, "Direct validation should fail for text file"
        assert not result_subprocess.is_valid, "Subprocess validation should fail for text file"
        
        print("\n✅ Both methods correctly identified text file as invalid NetCDF!")
            
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print(f"Cleaned up text file: {tmp_path}")

def test_timeout():
    """Test timeout functionality."""
    print("\n" + "="*50)
    print("Testing timeout functionality...")
    
    # Test with very short timeout on non-existent file
    print("Testing with 0.001 second timeout on non-existent file...")
    result = validate_GEOS5FP_NetCDF_file(
        "nonexistent.nc", 
        use_subprocess=True, 
        timeout_seconds=0.001
    )
    
    print(f"Timeout result - Valid: {result.is_valid}")
    print(f"Timeout result - Error: {result.error}")
    
    print("\n✅ Timeout test completed (may or may not have triggered timeout)")

if __name__ == "__main__":
    print("GEOS-5 FP Subprocess Validation Test (Simple)")
    print("=" * 60)
    
    success_count = 0
    total_tests = 0
    
    tests = [
        test_basic_functionality,
        test_empty_file,
        test_text_file
    ]
    
    for test_func in tests:
        total_tests += 1
        try:
            test_func()
            success_count += 1
        except Exception as e:
            print(f"\n❌ Test {test_func.__name__} failed with error: {e}")
    
    test_timeout()
    
    print("\n" + "=" * 60)
    print(f"Test Summary: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Subprocess isolation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the output above.")