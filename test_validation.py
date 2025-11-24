"""
Test script for GEOS-5 FP NetCDF file validation functionality.

This script demonstrates how to use the validate_GEOS5FP_NetCDF_file module
and tests its functionality with various scenarios.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the path to import GEOS5FP
sys.path.insert(0, str(Path(__file__).parent))

from GEOS5FP import (
    validate_GEOS5FP_NetCDF_file,
    validate_GEOS5FP_directory, 
    is_valid_GEOS5FP_file,
    quick_validate,
    GEOS5FPValidationError,
    GEOS5FPValidationResult,
    get_validation_summary
)


def test_validation_functions():
    """Test the validation functions with various scenarios."""
    
    print("GEOS-5 FP NetCDF File Validation Test")
    print("=" * 50)
    
    # Test 1: Non-existent file
    print("\n1. Testing non-existent file...")
    result = validate_GEOS5FP_NetCDF_file("non_existent_file.nc4", verbose=True)
    print(f"Result: {result.is_valid}")
    print(f"Errors: {result.errors}")
    
    # Test 2: Empty file
    print("\n2. Testing empty file...")
    with tempfile.NamedTemporaryFile(suffix=".nc4", delete=False) as tmp:
        empty_file = tmp.name
    
    try:
        result = validate_GEOS5FP_NetCDF_file(empty_file, verbose=True)
        print(f"Result: {result.is_valid}")
        print(f"Errors: {result.errors}")
    finally:
        os.unlink(empty_file)
    
    # Test 3: Invalid filename format
    print("\n3. Testing file with invalid filename format...")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"fake content")
        invalid_name_file = tmp.name
    
    try:
        result = validate_GEOS5FP_NetCDF_file(invalid_name_file, verbose=True)
        print(f"Result: {result.is_valid}")
        print(f"Warnings: {result.warnings}")
    finally:
        os.unlink(invalid_name_file)
    
    # Test 4: Properly named file (but fake content)
    print("\n4. Testing properly named file with fake content...")
    fake_geos5fp_name = "GEOS.fp.asm.tavg1_2d_slv_Nx.20250214_1200.V01.nc4"
    with tempfile.NamedTemporaryFile(suffix=".nc4", delete=False) as tmp:
        tmp.write(b"fake netcdf content")
        fake_file = tmp.name
    
    # Rename to proper GEOS5FP format
    proper_name_file = os.path.join(os.path.dirname(fake_file), fake_geos5fp_name)
    os.rename(fake_file, proper_name_file)
    
    try:
        result = validate_GEOS5FP_NetCDF_file(proper_name_file, verbose=True)
        print(f"Result: {result.is_valid}")
        print(f"Metadata: {result.metadata}")
        print(f"Errors: {result.errors}")
    finally:
        if os.path.exists(proper_name_file):
            os.unlink(proper_name_file)
    
    # Test 5: Quick validation
    print("\n5. Testing quick validation...")
    quick_result = quick_validate("non_existent_file.nc4")
    print(f"Quick validation result: {quick_result.is_valid}")
    
    # Test 6: Boolean convenience function
    print("\n6. Testing boolean convenience function...")
    is_valid = is_valid_GEOS5FP_file("non_existent_file.nc4")
    print(f"is_valid_GEOS5FP_file result: {is_valid}")
    
    # Test 7: Validation result string representation
    print("\n7. Testing validation result string representation...")
    result = validate_GEOS5FP_NetCDF_file("non_existent_file.nc4")
    print("Validation Result String:")
    print(result)
    
    print("\n" + "=" * 50)
    print("Test completed successfully!")


def test_directory_validation():
    """Test directory validation functionality."""
    
    print("\n" + "=" * 50)
    print("Testing directory validation...")
    
    # Create a temporary directory with some test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some fake GEOS5FP files
        file1 = os.path.join(temp_dir, "GEOS.fp.asm.tavg1_2d_slv_Nx.20250214_1200.V01.nc4")
        file2 = os.path.join(temp_dir, "GEOS.fp.asm.tavg1_2d_rad_Nx.20250214_1300.V01.nc4")
        file3 = os.path.join(temp_dir, "invalid_file.txt")
        
        # Create files with some content
        for filepath in [file1, file2]:
            with open(filepath, 'wb') as f:
                f.write(b"fake netcdf content")
        
        with open(file3, 'w') as f:
            f.write("not a netcdf file")
        
        # Test directory validation
        results = validate_GEOS5FP_directory(temp_dir, pattern="*.nc4", verbose=True)
        
        print(f"\nDirectory validation results:")
        for filename, result in results.items():
            print(f"  {filename}: {'VALID' if result.is_valid else 'INVALID'}")
        
        # Test summary
        summary = get_validation_summary(results)
        print(f"\nValidation Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")


def demonstrate_usage():
    """Demonstrate typical usage patterns."""
    
    print("\n" + "=" * 50)
    print("Usage Examples:")
    print("=" * 50)
    
    print("""
# Basic validation
from GEOS5FP import validate_GEOS5FP_NetCDF_file

result = validate_GEOS5FP_NetCDF_file("GEOS.fp.asm.tavg1_2d_slv_Nx.20250214_1200.V01.nc4")
if result.is_valid:
    print("File is valid!")
    print(f"File size: {result.metadata.get('file_size_mb', 'unknown')} MB")
else:
    print("File validation failed:")
    for error in result.errors:
        print(f"  - {error}")

# Quick boolean check
from GEOS5FP import is_valid_GEOS5FP_file

if is_valid_GEOS5FP_file("my_file.nc4"):
    print("File is valid!")

# Validate entire directory
from GEOS5FP import validate_GEOS5FP_directory, get_validation_summary

results = validate_GEOS5FP_directory("/path/to/geos5fp/files")
summary = get_validation_summary(results)
print(f"Validated {summary['total_files']} files, {summary['valid_files']} are valid")

# Custom validation with specific requirements
result = validate_GEOS5FP_NetCDF_file(
    "my_file.nc4",
    required_variables=["T2M", "QV2M"],  # Require specific variables
    min_file_size_mb=1.0,  # Minimum 1 MB
    check_data_integrity=True,  # Check data quality
    verbose=True  # Detailed logging
)
""")


if __name__ == "__main__":
    test_validation_functions()
    test_directory_validation()
    demonstrate_usage()