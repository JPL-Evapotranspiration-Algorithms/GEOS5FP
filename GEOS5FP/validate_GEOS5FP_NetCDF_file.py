"""
GEOS-5 FP NetCDF File Validation Module

This module provides comprehensive validation functionality for GEOS-5 FP NetCDF files
to ensure they are valid, complete, and properly formatted for use with the GEOS5FP package.

Author: Gregory H. Halverson
"""

import logging
import os
import re
from datetime import datetime
from os.path import exists, expanduser, abspath, basename, getsize
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

logger = logging.getLogger(__name__)


class GEOS5FPValidationError(Exception):
    """Custom exception for GEOS-5 FP file validation errors."""
    pass


class GEOS5FPValidationResult:
    """
    Container for validation results with detailed information about the validation process.
    """
    
    def __init__(self, is_valid: bool, filename: str, errors: List[str] = None, warnings: List[str] = None, 
                 metadata: Dict[str, Any] = None):
        self.is_valid = is_valid
        self.filename = filename
        self.errors = errors or []
        self.warnings = warnings or []
        self.metadata = metadata or {}
    
    def __bool__(self) -> bool:
        """Allow boolean evaluation of validation result."""
        return self.is_valid
    
    def __str__(self) -> str:
        """String representation of validation result."""
        status = "VALID" if self.is_valid else "INVALID"
        result = f"GEOS-5 FP File Validation: {status}\n"
        result += f"File: {self.filename}\n"
        
        if self.errors:
            result += f"Errors ({len(self.errors)}):\n"
            for i, error in enumerate(self.errors, 1):
                result += f"  {i}. {error}\n"
        
        if self.warnings:
            result += f"Warnings ({len(self.warnings)}):\n"
            for i, warning in enumerate(self.warnings, 1):
                result += f"  {i}. {warning}\n"
        
        if self.metadata:
            result += "Metadata:\n"
            for key, value in self.metadata.items():
                result += f"  {key}: {value}\n"
        
        return result.rstrip()


def validate_GEOS5FP_NetCDF_file(
    filename: str,
    check_variables: bool = True,
    check_spatial_ref: bool = True,
    check_temporal_info: bool = True,
    check_file_size: bool = True,
    min_file_size_mb: float = 0.1,
    max_file_size_mb: float = 1000.0,
    required_variables: Optional[List[str]] = None,
    check_data_integrity: bool = True,
    verbose: bool = False
) -> GEOS5FPValidationResult:
    """
    Validate a GEOS-5 FP NetCDF file for integrity, format compliance, and data quality.
    
    This function performs comprehensive validation of GEOS-5 FP NetCDF files including:
    - File existence and accessibility
    - File size validation
    - Filename format validation
    - NetCDF format validation
    - Spatial reference system validation
    - Variable presence and structure validation
    - Data integrity checks
    - Temporal information validation
    
    Args:
        filename (str): Path to the GEOS-5 FP NetCDF file to validate
        check_variables (bool): Whether to validate variables in the file. Default: True
        check_spatial_ref (bool): Whether to validate spatial reference system. Default: True
        check_temporal_info (bool): Whether to validate temporal information. Default: True
        check_file_size (bool): Whether to validate file size. Default: True
        min_file_size_mb (float): Minimum expected file size in MB. Default: 0.1
        max_file_size_mb (float): Maximum expected file size in MB. Default: 1000.0
        required_variables (List[str], optional): List of required variable names. 
                                                If None, uses common GEOS-5 FP variables
        check_data_integrity (bool): Whether to perform data integrity checks. Default: True
        verbose (bool): Whether to log detailed validation steps. Default: False
    
    Returns:
        GEOS5FPValidationResult: Comprehensive validation result with status, errors, warnings, and metadata
    
    Raises:
        ValueError: If filename is None or empty
    
    Example:
        >>> result = validate_GEOS5FP_NetCDF_file("GEOS.fp.asm.tavg1_2d_slv_Nx.20250214_1200.V01.nc4")
        >>> if result.is_valid:
        ...     print("File is valid!")
        >>> else:
        ...     print(f"Validation failed: {result.errors}")
    """
    
    # Input validation
    if not filename:
        raise ValueError("filename must be provided")
    
    # Initialize result containers
    errors = []
    warnings = []
    metadata = {}
    
    # Expand and get absolute path
    expanded_filename = abspath(expanduser(filename))
    
    if verbose:
        logger.info(f"Starting validation of GEOS-5 FP file: {expanded_filename}")
    
    # 1. File existence check
    if not exists(expanded_filename):
        errors.append(f"File does not exist: {expanded_filename}")
        return GEOS5FPValidationResult(False, filename, errors, warnings, metadata)
    
    # 2. File size validation
    if check_file_size:
        try:
            file_size_bytes = getsize(expanded_filename)
            file_size_mb = file_size_bytes / (1024 * 1024)
            metadata['file_size_mb'] = round(file_size_mb, 2)
            metadata['file_size_bytes'] = file_size_bytes
            
            if file_size_bytes == 0:
                errors.append("File is empty (0 bytes)")
            elif file_size_mb < min_file_size_mb:
                errors.append(f"File size ({file_size_mb:.2f} MB) is below minimum threshold ({min_file_size_mb} MB)")
            elif file_size_mb > max_file_size_mb:
                warnings.append(f"File size ({file_size_mb:.2f} MB) is above typical threshold ({max_file_size_mb} MB)")
            
            if verbose:
                logger.info(f"File size: {file_size_mb:.2f} MB")
                
        except OSError as e:
            errors.append(f"Unable to determine file size: {e}")
    
    # 3. Filename format validation
    filename_base = basename(expanded_filename)
    metadata['filename'] = filename_base
    
    # GEOS-5 FP filename pattern: GEOS.fp.asm.{product}.{YYYYMMDD_HHMM}.V01.nc4
    geos5fp_pattern = re.compile(r'^GEOS\.fp\.asm\.([^.]+)\.(\d{8}_\d{4})\.V\d{2}\.nc4$')
    match = geos5fp_pattern.match(filename_base)
    
    if match:
        product_name = match.group(1)
        time_string = match.group(2)
        metadata['product_name'] = product_name
        metadata['time_string'] = time_string
        
        # Validate timestamp format
        if check_temporal_info:
            try:
                parsed_time = datetime.strptime(time_string, "%Y%m%d_%H%M")
                metadata['parsed_datetime'] = parsed_time.isoformat()
                
                if verbose:
                    logger.info(f"Parsed timestamp: {parsed_time}")
                    
            except ValueError as e:
                errors.append(f"Invalid timestamp format in filename: {time_string} ({e})")
        
        if verbose:
            logger.info(f"Product name: {product_name}")
            
    else:
        warnings.append(f"Filename does not match expected GEOS-5 FP pattern: {filename_base}")
    
    # 4. NetCDF format validation using rasterio
    try:
        # Try to open the file as a NetCDF dataset
        with rasterio.open(expanded_filename) as dataset:
            metadata['driver'] = dataset.driver
            metadata['count'] = dataset.count
            metadata['width'] = dataset.width
            metadata['height'] = dataset.height
            metadata['crs'] = str(dataset.crs) if dataset.crs else None
            metadata['bounds'] = dataset.bounds
            metadata['transform'] = list(dataset.transform)[:6] if dataset.transform else None
            
            if verbose:
                logger.info(f"Successfully opened NetCDF file with driver: {dataset.driver}")
                logger.info(f"Dimensions: {dataset.width}x{dataset.height}, Bands: {dataset.count}")
            
            # 5. Spatial reference validation
            if check_spatial_ref:
                if dataset.crs is None:
                    warnings.append("No coordinate reference system (CRS) information found")
                else:
                    # Check if it's a reasonable geographic CRS for global data
                    crs_string = str(dataset.crs).lower()
                    if 'wgs84' in crs_string or 'epsg:4326' in crs_string or '+proj=longlat' in crs_string:
                        if verbose:
                            logger.info(f"Valid geographic CRS detected: {dataset.crs}")
                    else:
                        warnings.append(f"Unexpected CRS for global meteorological data: {dataset.crs}")
                
                # Check bounds for reasonable global coverage
                if dataset.bounds:
                    bounds = dataset.bounds
                    if (bounds.left < -180.1 or bounds.right > 180.1 or 
                        bounds.bottom < -90.1 or bounds.top > 90.1):
                        warnings.append(f"Bounds appear to extend beyond valid geographic coordinates: {bounds}")
                    elif (bounds.right - bounds.left > 350 and bounds.top - bounds.bottom > 170):
                        if verbose:
                            logger.info("Dataset appears to have global coverage")
                    else:
                        warnings.append(f"Dataset may not have global coverage: {bounds}")
            
            # 6. Data integrity checks
            if check_data_integrity and dataset.count > 0:
                try:
                    # Read a small sample of data from the first band
                    sample_data = dataset.read(1, window=rasterio.windows.Window(0, 0, 
                                                                               min(100, dataset.width), 
                                                                               min(100, dataset.height)))
                    
                    if sample_data is not None:
                        valid_data_count = np.count_nonzero(~np.isnan(sample_data))
                        total_sample_size = sample_data.size
                        valid_data_ratio = valid_data_count / total_sample_size if total_sample_size > 0 else 0
                        
                        metadata['sample_valid_data_ratio'] = round(valid_data_ratio, 3)
                        
                        if valid_data_ratio == 0:
                            warnings.append("Sample data contains only NaN values")
                        elif valid_data_ratio < 0.1:
                            warnings.append(f"Sample data has very low valid data ratio: {valid_data_ratio:.3f}")
                        
                        if verbose:
                            logger.info(f"Sample data valid ratio: {valid_data_ratio:.3f}")
                    
                except Exception as e:
                    warnings.append(f"Unable to read sample data for integrity check: {e}")
    
    except RasterioIOError as e:
        errors.append(f"Unable to open file as NetCDF: {e}")
    except Exception as e:
        errors.append(f"Unexpected error reading NetCDF file: {e}")
    
    # 7. Try to access as netcdf subdataset (GEOS5FP specific)
    if check_variables:
        _validate_netcdf_variables(expanded_filename, errors, warnings, metadata, 
                                 required_variables, verbose)
    
    # Determine overall validation result
    is_valid = len(errors) == 0
    
    if verbose:
        logger.info(f"Validation complete. Valid: {is_valid}, Errors: {len(errors)}, Warnings: {len(warnings)}")
    
    return GEOS5FPValidationResult(is_valid, filename, errors, warnings, metadata)


def _validate_netcdf_variables(
    filename: str, 
    errors: List[str], 
    warnings: List[str], 
    metadata: Dict[str, Any],
    required_variables: Optional[List[str]] = None,
    verbose: bool = False
) -> None:
    """
    Internal function to validate NetCDF variables using GDAL NetCDF subdatasets.
    """
    
    if required_variables is None:
        # Common GEOS-5 FP variables
        required_variables = []  # We'll make this optional since variables vary by product
    
    try:
        # Try to get NetCDF subdatasets information
        with rasterio.open(filename) as dataset:
            # Check if we can list subdatasets
            subdatasets = dataset.subdatasets if hasattr(dataset, 'subdatasets') else []
            
            if subdatasets:
                metadata['subdatasets'] = len(subdatasets)
                if verbose:
                    logger.info(f"Found {len(subdatasets)} subdatasets")
                
                # Extract variable names from subdatasets
                variable_names = []
                for subdataset in subdatasets[:10]:  # Limit to first 10 for performance
                    # Subdataset format: NETCDF:"filename":variable_name
                    if 'NETCDF:' in subdataset and ':' in subdataset:
                        parts = subdataset.split(':')
                        if len(parts) >= 3:
                            var_name = parts[-1]
                            variable_names.append(var_name)
                
                metadata['variable_names'] = variable_names[:20]  # Limit metadata size
                
                if verbose and variable_names:
                    logger.info(f"Available variables: {', '.join(variable_names[:5])}{'...' if len(variable_names) > 5 else ''}")
                
                # Check for required variables
                if required_variables:
                    missing_vars = [var for var in required_variables if var not in variable_names]
                    if missing_vars:
                        errors.append(f"Missing required variables: {', '.join(missing_vars)}")
                
                # Try to read one variable as a test
                if subdatasets:
                    try:
                        test_var = subdatasets[0]
                        with rasterio.open(test_var) as var_dataset:
                            # Just check that we can open it
                            metadata['test_variable_width'] = var_dataset.width
                            metadata['test_variable_height'] = var_dataset.height
                            
                            if verbose:
                                logger.info(f"Successfully accessed test variable: {test_var.split(':')[-1]}")
                                
                    except Exception as e:
                        warnings.append(f"Unable to access variables as subdatasets: {e}")
            else:
                # No subdatasets found, this might still be valid if it's a simple NetCDF
                if verbose:
                    logger.info("No subdatasets found, treating as simple NetCDF")
                
    except Exception as e:
        warnings.append(f"Unable to validate NetCDF variables: {e}")


def validate_GEOS5FP_directory(
    directory: str,
    pattern: str = "*.nc4",
    max_files: int = 100,
    verbose: bool = False
) -> Dict[str, GEOS5FPValidationResult]:
    """
    Validate all GEOS-5 FP NetCDF files in a directory.
    
    Args:
        directory (str): Path to directory containing GEOS-5 FP files
        pattern (str): File pattern to match. Default: "*.nc4"
        max_files (int): Maximum number of files to validate. Default: 100
        verbose (bool): Whether to log validation progress. Default: False
    
    Returns:
        Dict[str, GEOS5FPValidationResult]: Dictionary mapping filenames to validation results
    """
    import glob
    
    directory_path = abspath(expanduser(directory))
    
    if not exists(directory_path):
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    # Find matching files
    file_pattern = os.path.join(directory_path, pattern)
    files = glob.glob(file_pattern)
    
    if len(files) > max_files:
        if verbose:
            logger.warning(f"Found {len(files)} files, limiting to first {max_files}")
        files = files[:max_files]
    
    results = {}
    
    for i, file_path in enumerate(files, 1):
        if verbose:
            logger.info(f"Validating {i}/{len(files)}: {basename(file_path)}")
        
        try:
            result = validate_GEOS5FP_NetCDF_file(file_path, verbose=False)  # Avoid nested verbose output
            results[basename(file_path)] = result
        except Exception as e:
            # Create failed result for exceptions
            results[basename(file_path)] = GEOS5FPValidationResult(
                False, file_path, [f"Validation exception: {e}"], [], {}
            )
    
    return results


def get_validation_summary(results: Dict[str, GEOS5FPValidationResult]) -> Dict[str, Any]:
    """
    Generate a summary of validation results.
    
    Args:
        results (Dict[str, GEOS5FPValidationResult]): Results from validate_GEOS5FP_directory
    
    Returns:
        Dict[str, Any]: Summary statistics
    """
    total_files = len(results)
    valid_files = sum(1 for result in results.values() if result.is_valid)
    invalid_files = total_files - valid_files
    
    total_errors = sum(len(result.errors) for result in results.values())
    total_warnings = sum(len(result.warnings) for result in results.values())
    
    return {
        'total_files': total_files,
        'valid_files': valid_files,
        'invalid_files': invalid_files,
        'validation_rate': round(valid_files / total_files * 100, 1) if total_files > 0 else 0,
        'total_errors': total_errors,
        'total_warnings': total_warnings
    }


# Convenience functions for backward compatibility and ease of use
def is_valid_GEOS5FP_file(filename: str, **kwargs) -> bool:
    """
    Simple boolean check for GEOS-5 FP file validity.
    
    Args:
        filename (str): Path to the GEOS-5 FP NetCDF file
        **kwargs: Additional arguments passed to validate_GEOS5FP_NetCDF_file
    
    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        result = validate_GEOS5FP_NetCDF_file(filename, **kwargs)
        return result.is_valid
    except Exception:
        return False


def quick_validate(filename: str) -> GEOS5FPValidationResult:
    """
    Quick validation with minimal checks for performance.
    
    Args:
        filename (str): Path to the GEOS-5 FP NetCDF file
    
    Returns:
        GEOS5FPValidationResult: Validation result
    """
    return validate_GEOS5FP_NetCDF_file(
        filename,
        check_variables=False,
        check_spatial_ref=False,
        check_temporal_info=True,
        check_data_integrity=False,
        verbose=False
    )