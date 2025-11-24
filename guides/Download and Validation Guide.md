# Download and Validation Guide

## Overview

The GEOS5FP library includes a comprehensive validation system for NetCDF files and integrates it into the download process to create a robust, self-healing download system. This ensures that downloaded files are valid and complete before use, with automatic cleanup and retry for corrupted downloads.

## Features

### File Validation

- **File Existence & Size Validation**: Checks if files exist and have reasonable sizes
- **Filename Pattern Validation**: Validates GEOS-5 FP naming conventions
- **NetCDF Format Validation**: Ensures files are valid NetCDF format and can be opened
- **Spatial Reference Validation**: Checks coordinate reference systems and bounds
- **Variable Validation**: Validates NetCDF variables and subdatasets
- **Data Integrity Checks**: Samples data to check for corruption or invalid values
- **Temporal Information Parsing**: Extracts and validates timestamps from filenames
- **Batch Directory Validation**: Validates multiple files efficiently
- **Detailed Error Reporting**: Provides comprehensive error and warning messages

### Download Integration

- **Smart File Reuse**: Validates existing files before re-downloading
- **Robust Download Process**: Automatic retry with validation after each attempt
- **Automatic Cleanup**: Removes invalid or corrupted files automatically
- **Detailed Logging**: Provides visibility into validation results

## Validation Module

### Core Function: `validate_GEOS5FP_NetCDF_file()`

The main validation function performs comprehensive checks on GEOS-5 FP NetCDF files.

```python
from GEOS5FP import validate_GEOS5FP_NetCDF_file

# Validate a single file
result = validate_GEOS5FP_NetCDF_file("GEOS.fp.asm.tavg1_2d_slv_Nx.20250214_1200.V01.nc4")

if result.is_valid:
    print("File is valid!")
    print(f"File size: {result.metadata.get('file_size_mb')} MB")
    print(f"Product: {result.metadata.get('product_name')}")
else:
    print("Validation failed:")
    for error in result.errors:
        print(f"  - {error}")
```

### Validation Result: `GEOS5FPValidationResult`

The validation function returns a structured result object with:

- **`is_valid`**: Boolean indicating overall validation status
- **`filename`**: Path to the validated file
- **`errors`**: List of error messages (validation fails if any errors exist)
- **`warnings`**: List of warning messages (non-fatal issues)
- **`metadata`**: Dictionary with extracted file information:
  - `file_size_mb`: File size in megabytes
  - `product_name`: GEOS-5 FP product name
  - `time_string`: Timestamp string from filename
  - `parsed_datetime`: Parsed datetime in ISO format
  - `driver`: Rasterio driver used to open file
  - `width`, `height`: Raster dimensions
  - `crs`: Coordinate reference system
  - `bounds`: Spatial bounds
  - `variable_names`: List of available variables

### Validation Checks Performed

1. **File Existence**: Verifies the file exists and is accessible
2. **File Size**: Ensures file size is within reasonable bounds (0.1 MB - 1000 MB by default)
3. **Filename Format**: Validates against GEOS-5 FP naming pattern: `GEOS.fp.asm.{product}.{YYYYMMDD_HHMM}.V{version}.nc4`
4. **NetCDF Format**: Attempts to open file using rasterio/GDAL NetCDF driver
5. **Coordinate System**: Checks for valid geographic CRS (WGS84/EPSG:4326 expected)
6. **Global Coverage**: Validates bounds are reasonable for global meteorological data
7. **Variables**: Lists available NetCDF variables and checks for required ones
8. **Data Integrity**: Samples data to check for corruption, NaN values, or read errors
9. **Temporal Parsing**: Extracts and validates timestamp from filename

## Usage Examples

### Quick Boolean Check

```python
from GEOS5FP import is_valid_GEOS5FP_file

if is_valid_GEOS5FP_file("my_file.nc4"):
    print("File is valid!")
else:
    print("File is invalid")
```

### Custom Validation Options

```python
result = validate_GEOS5FP_NetCDF_file(
    "my_file.nc4",
    required_variables=["T2M", "QV2M"],  # Require specific variables
    min_file_size_mb=1.0,                # Minimum file size
    max_file_size_mb=100.0,              # Maximum file size  
    check_data_integrity=True,           # Check data quality
    check_spatial_ref=True,              # Validate CRS and bounds
    verbose=True                         # Detailed logging
)
```

### Directory Validation

Validate all GEOS-5 FP files in a directory:

```python
from GEOS5FP import validate_GEOS5FP_directory, get_validation_summary

# Validate all .nc4 files in a directory
results = validate_GEOS5FP_directory("/path/to/geos5fp/data")

# Get summary statistics
summary = get_validation_summary(results)
print(f"Validated {summary['total_files']} files")
print(f"Valid: {summary['valid_files']} ({summary['validation_rate']}%)")
print(f"Errors: {summary['total_errors']}")
```

### Batch Validation with Filtering

```python
from GEOS5FP import validate_GEOS5FP_directory

# Validate directory
results = validate_GEOS5FP_directory("~/data/GEOS5FP")

# Find all valid files
valid_files = [name for name, result in results.items() if result.is_valid]
print(f"Valid files: {valid_files}")

# Find all files with errors
invalid_files = {name: result.errors for name, result in results.items() if not result.is_valid}
for name, errors in invalid_files.items():
    print(f"\n{name}:")
    for error in errors:
        print(f"  - {error}")
```

## Enhanced Download System

### How It Works

The download system integrates validation at two points:

#### 1. Pre-Download Validation

Before downloading, the system checks if the file already exists:

```python
# Check if file already exists and is valid
if exists(expanded_filename):
    validation_result = validate_GEOS5FP_NetCDF_file(expanded_filename)
    if validation_result.is_valid:
        return GEOS5FPGranule(filename)  # Reuse existing valid file
    else:
        os.remove(expanded_filename)     # Clean up invalid file
```

#### 2. Post-Download Validation with Retry

After downloading, the system validates the file and retries if invalid:

```python
# Validate downloaded file and retry if invalid
validation_result = validate_GEOS5FP_NetCDF_file(expanded_filename)
if validation_result.is_valid:
    return GEOS5FPGranule(result_filename)
else:
    os.remove(expanded_filename)  # Clean up invalid download
    # Retry download if attempts remaining
```

### Automatic Usage

The enhanced download system works automatically with all GEOS5FP data retrieval methods. No code changes are required:

```python
from GEOS5FP import GEOS5FP
from sentinel_tiles import sentinel_tiles

# Existing code works exactly the same
geos5fp = GEOS5FP()
geometry = sentinel_tiles.grid("11SPS")
timestamp = "2025-02-22 12:00:00"

# These now automatically include validation
Ta_C = geos5fp.Ta_C(time_UTC=timestamp, geometry=geometry)
SM = geos5fp.SM(time_UTC=timestamp, geometry=geometry)
```

### Integration with Custom Downloads

You can also use validation in your own download functions:

```python
from GEOS5FP import validate_GEOS5FP_NetCDF_file
from GEOS5FP.download_file import download_file
import os

def download_with_validation(url, filename, max_retries=3):
    """Download and validate GEOS-5 FP file with retry logic."""
    
    for attempt in range(max_retries):
        try:
            # Download the file
            print(f"Download attempt {attempt + 1}/{max_retries}")
            downloaded_file = download_file(url, filename)
            
            # Validate the downloaded file
            result = validate_GEOS5FP_NetCDF_file(downloaded_file)
            
            if result.is_valid:
                print(f"Successfully downloaded and validated: {filename}")
                print(f"  Size: {result.metadata['file_size_mb']:.2f} MB")
                print(f"  Product: {result.metadata['product_name']}")
                return downloaded_file
            else:
                print(f"Downloaded file failed validation: {filename}")
                for error in result.errors:
                    print(f"  Error: {error}")
                
                # Remove invalid file
                if os.path.exists(downloaded_file):
                    os.remove(downloaded_file)
                    print(f"  Removed invalid file")
                
                if attempt < max_retries - 1:
                    print("  Retrying...")
                    
        except Exception as e:
            print(f"Download failed: {e}")
            if attempt < max_retries - 1:
                print("  Retrying...")
    
    raise Exception(f"Failed to download valid file after {max_retries} attempts")
```

## Benefits

### Performance Improvements

- **Faster Execution**: Reuses valid existing files instead of re-downloading
- **Reduced Bandwidth**: Only downloads when necessary
- **Efficient Caching**: Validates cached files before use

### Reliability Improvements

- **Self-Healing**: Automatically removes corrupted files
- **Retry Logic**: Handles transient network issues
- **Quality Assurance**: Ensures data integrity before processing

### Enhanced Diagnostics

- **Detailed Logging**: Shows validation process and results
- **Error Categorization**: Specific error types for different issues
- **Metadata Extraction**: Rich information about files and data

## Log Output Examples

### Successful Existing File Reuse

```
INFO - checking existing file: ~/data/GEOS5FP/2025.02.22/GEOS.fp.asm.tavg1_2d_slv_Nx.20250222_1200.V01.nc4
INFO - existing file is valid: ~/data/GEOS5FP/2025.02.22/GEOS.fp.asm.tavg1_2d_slv_Nx.20250222_1200.V01.nc4 (15.2 MB)
```

### Invalid File Cleanup and Download

```
WARNING - existing file is invalid, removing: ~/data/GEOS5FP/2025.02.22/GEOS.fp.asm.tavg1_2d_slv_Nx.20250222_1200.V01.nc4
WARNING -   validation error: File size (0.00 MB) is below minimum threshold (0.1 MB)
INFO - removed invalid file: ~/data/GEOS5FP/2025.02.22/GEOS.fp.asm.tavg1_2d_slv_Nx.20250222_1200.V01.nc4
INFO - download attempt 1/3: https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y2025/M02/D22/GEOS.fp.asm.tavg1_2d_slv_Nx.20250222_1200.V01.nc4
INFO - validating downloaded file: ~/data/GEOS5FP/2025.02.22/GEOS.fp.asm.tavg1_2d_slv_Nx.20250222_1200.V01.nc4
INFO - download and validation successful: ~/data/GEOS5FP/2025.02.22/GEOS.fp.asm.tavg1_2d_slv_Nx.20250222_1200.V01.nc4 (15.2 MB)
INFO - validated product: tavg1_2d_slv_Nx
```

## Common Use Cases

### Quality Assurance

Run validation on downloaded files to ensure they're not corrupted before processing.

```python
from GEOS5FP import validate_GEOS5FP_NetCDF_file

files_to_check = ["file1.nc4", "file2.nc4", "file3.nc4"]

for filename in files_to_check:
    result = validate_GEOS5FP_NetCDF_file(filename)
    if not result.is_valid:
        print(f"Problem with {filename}:")
        for error in result.errors:
            print(f"  - {error}")
```

### Batch Processing

Validate entire directories of GEOS-5 FP data to identify problematic files before processing.

```python
from GEOS5FP import validate_GEOS5FP_directory, get_validation_summary

# Validate all files
results = validate_GEOS5FP_directory("~/data/GEOS5FP")
summary = get_validation_summary(results)

# Get list of valid files for processing
valid_files = [name for name, result in results.items() if result.is_valid]

# Process only valid files
for filename in valid_files:
    # Your processing code here
    pass
```

### Download Verification

Integrate with download functions to automatically retry failed downloads.

```python
from GEOS5FP import validate_GEOS5FP_NetCDF_file
import requests

def download_and_verify(url, filename, max_retries=3):
    for attempt in range(max_retries):
        # Download file
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        # Verify download
        result = validate_GEOS5FP_NetCDF_file(filename)
        if result.is_valid:
            return filename
        else:
            print(f"Attempt {attempt + 1} failed validation, retrying...")
            os.remove(filename)
    
    raise Exception("Download failed after all retries")
```

### Data Pipeline Integration

Use as a preprocessing step in automated data processing pipelines.

```python
from GEOS5FP import GEOS5FP, is_valid_GEOS5FP_file
from datetime import datetime

def process_geos5fp_data(timestamp, geometry):
    """Data processing pipeline with validation."""
    
    geos5fp = GEOS5FP()
    
    # Download data (includes automatic validation)
    temp = geos5fp.Ta_C(time_UTC=timestamp, geometry=geometry)
    
    # The file path is stored in the result
    if hasattr(temp, 'filename'):
        # Double-check validation if needed
        if not is_valid_GEOS5FP_file(temp.filename):
            raise Exception("Downloaded file failed validation")
    
    # Continue with processing...
    return temp
```

## Error Handling

The validation function is designed to be robust and handle various error conditions gracefully. All errors are captured and reported in the validation result rather than raising exceptions.

### Common Error Types

- **File not found**: Non-existent files
- **Empty file**: Zero-size files
- **Corrupted NetCDF**: Files that cannot be opened with NetCDF driver
- **Invalid naming**: Files with incorrect GEOS-5 FP naming pattern
- **Size issues**: Files too small or too large
- **Network issues**: Problems with network-mounted file access
- **Permission errors**: Insufficient permissions to read file

### Example Error Handling

```python
from GEOS5FP import validate_GEOS5FP_NetCDF_file

result = validate_GEOS5FP_NetCDF_file("my_file.nc4")

if not result.is_valid:
    # Handle different error types
    if "does not exist" in str(result.errors):
        print("File not found - need to download")
    elif "File size" in str(result.errors):
        print("File size issue - might be incomplete download")
    elif "NetCDF" in str(result.errors):
        print("File is corrupted - need to re-download")
    else:
        print(f"Other validation issues: {result.errors}")
```

## File Structure

```
GEOS5FP/
├── validate_GEOS5FP_NetCDF_file.py  # Main validation module
├── GEOS5FP_connection.py            # Enhanced download method
└── __init__.py                       # Exports validation functions

tests/
└── test_validate_GEOS5FP_NetCDF_file.py  # Comprehensive test suite
```

## Testing

The validation system includes comprehensive tests covering:

- File existence and accessibility
- File size validation
- Filename pattern matching
- NetCDF format validation
- Spatial reference checking
- Variable validation
- Data integrity checks
- Directory batch validation
- Error handling scenarios

All tests pass and backward compatibility is maintained.

## Summary

The GEOS-5 FP download and validation system provides:

1. **Automatic Quality Assurance** - Files are validated before use
2. **Intelligent Caching** - Valid existing files are reused
3. **Self-Healing Capabilities** - Invalid files are automatically cleaned up
4. **Robust Error Handling** - Detailed diagnostics and retry logic
5. **Zero-Change Integration** - Works with all existing code
6. **Performance Optimization** - Faster execution through smart reuse
7. **Comprehensive Validation** - Multiple validation checks ensure data quality

The system is production-ready and significantly improves the reliability and performance of GEOS-5 FP data workflows.
