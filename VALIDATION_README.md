# GEOS-5 FP NetCDF File Validation

This module provides comprehensive validation functionality for GEOS-5 FP NetCDF files to ensure they are valid, complete, and properly formatted.

## Features

- **File Existence & Size Validation**: Checks if files exist and have reasonable sizes
- **Filename Pattern Validation**: Validates GEOS-5 FP naming conventions
- **NetCDF Format Validation**: Ensures files are valid NetCDF format and can be opened
- **Spatial Reference Validation**: Checks coordinate reference systems and bounds
- **Variable Validation**: Validates NetCDF variables and subdatasets
- **Data Integrity Checks**: Samples data to check for corruption or invalid values
- **Temporal Information Parsing**: Extracts and validates timestamps from filenames
- **Batch Directory Validation**: Validates multiple files efficiently
- **Detailed Error Reporting**: Provides comprehensive error and warning messages

## Usage

### Basic Validation

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

### Quick Boolean Check

```python
from GEOS5FP import is_valid_GEOS5FP_file

if is_valid_GEOS5FP_file("my_file.nc4"):
    print("File is valid!")
else:
    print("File is invalid")
```

### Directory Validation

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

### Integration with Downloads

```python
from GEOS5FP.download_file import download_file

def download_with_validation(url, filename):
    """Download and validate GEOS-5 FP file."""
    try:
        # Download the file
        downloaded_file = download_file(url, filename)
        
        # Validate the downloaded file
        result = validate_GEOS5FP_NetCDF_file(downloaded_file)
        
        if result.is_valid:
            print(f"Successfully downloaded and validated: {filename}")
            return downloaded_file
        else:
            print(f"Downloaded file failed validation: {filename}")
            for error in result.errors:
                print(f"  Error: {error}")
            
            # Remove invalid file
            os.remove(downloaded_file)
            raise Exception("Downloaded file is invalid")
            
    except Exception as e:
        print(f"Download/validation failed: {e}")
        raise
```

## Validation Checks

The validation function performs the following checks:

1. **File Existence**: Verifies the file exists and is accessible
2. **File Size**: Ensures file size is within reasonable bounds (0.1 MB - 1000 MB by default)
3. **Filename Format**: Validates against GEOS-5 FP naming pattern: `GEOS.fp.asm.{product}.{YYYYMMDD_HHMM}.V{version}.nc4`
4. **NetCDF Format**: Attempts to open file using rasterio/GDAL NetCDF driver
5. **Coordinate System**: Checks for valid geographic CRS (WGS84/EPSG:4326 expected)
6. **Global Coverage**: Validates bounds are reasonable for global meteorological data
7. **Variables**: Lists available NetCDF variables and checks for required ones
8. **Data Integrity**: Samples data to check for corruption, NaN values, or read errors
9. **Temporal Parsing**: Extracts and validates timestamp from filename

## Return Values

The `validate_GEOS5FP_NetCDF_file` function returns a `GEOS5FPValidationResult` object with:

- `is_valid`: Boolean indicating overall validation status
- `filename`: Path to the validated file
- `errors`: List of error messages (validation fails if any errors exist)
- `warnings`: List of warning messages (non-fatal issues)
- `metadata`: Dictionary with extracted file information:
  - `file_size_mb`: File size in megabytes
  - `product_name`: GEOS-5 FP product name
  - `time_string`: Timestamp string from filename
  - `parsed_datetime`: Parsed datetime in ISO format
  - `driver`: Rasterio driver used to open file
  - `width`, `height`: Raster dimensions
  - `crs`: Coordinate reference system
  - `bounds`: Spatial bounds
  - `variable_names`: List of available variables

## Common Use Cases

### Quality Assurance
Run validation on downloaded files to ensure they're not corrupted before processing.

### Batch Processing
Validate entire directories of GEOS-5 FP data to identify problematic files.

### Download Verification
Integrate with download functions to automatically retry failed downloads.

### Data Pipeline Integration
Use as a preprocessing step in automated data processing pipelines.

## Error Handling

The validation function is designed to be robust and handle various error conditions:

- Non-existent files
- Empty or zero-size files  
- Corrupted NetCDF files
- Files with invalid naming
- Network-mounted file access issues
- Permission errors

All errors are captured and reported in the validation result rather than raising exceptions.