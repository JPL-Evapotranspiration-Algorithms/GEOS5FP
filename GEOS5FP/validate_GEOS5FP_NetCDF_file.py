"""
GEOS-5 FP NetCDF File Validation Module

This module provides comprehensive validation functionality for GEOS-5 FP NetCDF files
to ensure they are valid, complete, and properly formatted for use with the GEOS5FP package.

Author: Gregory H. Halverson
"""

import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
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
    
    @property
    def error(self) -> Optional[str]:
        """Get the first error or None if no errors."""
        return self.errors[0] if self.errors else None
    
    @property
    def issues(self) -> List[str]:
        """Get combined list of errors and warnings."""
        return self.errors + self.warnings
    
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
    verbose: bool = False,
    use_subprocess: bool = True,
    timeout_seconds: int = 30
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
    
    The validation is resilient to low-level C++ crashes from GDAL/NetCDF libraries
    by using process isolation when needed.
    
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
        use_subprocess (bool): Whether to use subprocess isolation for safety. Default: True
        timeout_seconds (int): Timeout for subprocess validation. Default: 30
    
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
    
    # Use subprocess isolation for safety against C++ crashes
    if use_subprocess:
        return _validate_in_subprocess(
            filename=filename,
            check_variables=check_variables,
            check_spatial_ref=check_spatial_ref,
            check_temporal_info=check_temporal_info,
            check_file_size=check_file_size,
            min_file_size_mb=min_file_size_mb,
            max_file_size_mb=max_file_size_mb,
            required_variables=required_variables,
            check_data_integrity=check_data_integrity,
            verbose=verbose,
            timeout_seconds=timeout_seconds
        )
    
    # Direct validation (legacy mode, less safe)
    return _validate_direct(
        filename=filename,
        check_variables=check_variables,
        check_spatial_ref=check_spatial_ref,
        check_temporal_info=check_temporal_info,
        check_file_size=check_file_size,
        min_file_size_mb=min_file_size_mb,
        max_file_size_mb=max_file_size_mb,
        required_variables=required_variables,
        check_data_integrity=check_data_integrity,
        verbose=verbose
    )


def _validate_direct(
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
    Direct validation without subprocess isolation (legacy mode).
    """
    
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


def safe_validate_GEOS5FP_NetCDF_file(filename: str, **kwargs) -> GEOS5FPValidationResult:
    """
    Ultra-safe validation that always uses subprocess isolation.
    
    This function is specifically designed to handle files that might cause
    C++ crashes or hangs in GDAL/NetCDF libraries.
    
    Args:
        filename (str): Path to the GEOS-5 FP NetCDF file
        **kwargs: Additional arguments passed to validate_GEOS5FP_NetCDF_file
    
    Returns:
        GEOS5FPValidationResult: Validation result
    """
    kwargs['use_subprocess'] = True
    return validate_GEOS5FP_NetCDF_file(filename, **kwargs)


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
        verbose=False,
        use_subprocess=False  # Skip subprocess for quick validation
    )


def _validate_in_subprocess(
    filename: str,
    check_variables: bool = True,
    check_spatial_ref: bool = True,
    check_temporal_info: bool = True,
    check_file_size: bool = True,
    min_file_size_mb: float = 0.1,
    max_file_size_mb: float = 1000.0,
    required_variables: Optional[List[str]] = None,
    check_data_integrity: bool = True,
    verbose: bool = False,
    timeout_seconds: int = 30
) -> GEOS5FPValidationResult:
    """
    Validate file using subprocess isolation to protect against C++ crashes.
    """
    import json
    
    # Create temporary files for communication
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file:
        config = {
            'filename': filename,
            'check_variables': check_variables,
            'check_spatial_ref': check_spatial_ref,
            'check_temporal_info': check_temporal_info,
            'check_file_size': check_file_size,
            'min_file_size_mb': min_file_size_mb,
            'max_file_size_mb': max_file_size_mb,
            'required_variables': required_variables,
            'check_data_integrity': check_data_integrity,
            'verbose': verbose
        }
        json.dump(config, config_file)
        config_file_path = config_file.name
    
    result_file_path = config_file_path.replace('.json', '_result.json')
    
    try:
        # Create the subprocess validation script
        validator_script = _create_subprocess_validator_script()
        
        # Run validation in subprocess with timeout
        cmd = [sys.executable, '-c', validator_script, config_file_path, result_file_path]
        
        if verbose:
            logger.info(f"Running subprocess validation with timeout {timeout_seconds}s")
        
        try:
            result = subprocess.run(
                cmd,
                timeout=timeout_seconds,
                capture_output=True,
                text=True,
                check=False  # Don't raise on non-zero exit
            )
            
            # Check if result file was created
            if exists(result_file_path):
                with open(result_file_path, 'r') as f:
                    result_dict = json.load(f)
                
                # Convert dictionary to GEOS5FPValidationResult
                validation_result = GEOS5FPValidationResult(
                    result_dict['is_valid'],
                    result_dict['filename'],
                    result_dict['errors'],
                    result_dict['warnings'],
                    result_dict['metadata']
                )
                    
                if verbose:
                    logger.info("Subprocess validation completed successfully")
                    
                return validation_result
            else:
                # Process completed but no result file - something went wrong
                error_msg = f"Subprocess validation failed - no result file created"
                if result.stderr:
                    error_msg += f": {result.stderr.strip()}"
                
                if verbose:
                    logger.warning(error_msg)
                
                return GEOS5FPValidationResult(
                    False, filename, 
                    [f"Subprocess validation failed: {error_msg}"], 
                    [], {}
                )
                
        except subprocess.TimeoutExpired:
            error_msg = f"Validation timed out after {timeout_seconds} seconds - possible C++ crash or hang"
            if verbose:
                logger.warning(error_msg)
                
            return GEOS5FPValidationResult(
                False, filename,
                [error_msg],
                [], {}
            )
            
        except subprocess.SubprocessError as e:
            error_msg = f"Subprocess validation error: {e}"
            if verbose:
                logger.warning(error_msg)
                
            return GEOS5FPValidationResult(
                False, filename,
                [error_msg],
                [], {}
            )
            
    finally:
        # Clean up temporary files
        for temp_file in [config_file_path, result_file_path]:
            if exists(temp_file):
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass  # Ignore cleanup errors


def _create_subprocess_validator_script() -> str:
    """
    Create the Python script code for subprocess validation.
    """
    return '''
import sys
import json
import os
from os.path import exists, expanduser, abspath, basename, getsize
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any

# Suppress GDAL warnings to avoid cluttering output
os.environ['CPL_LOG'] = '/dev/null'
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'YES'

try:
    import numpy as np
    import rasterio
    from rasterio.errors import RasterioIOError
    
    # Validation result class (simplified for subprocess)
    class ValidationResult:
        def __init__(self, is_valid, filename, errors=None, warnings=None, metadata=None):
            self.is_valid = is_valid
            self.filename = filename
            self.errors = errors or []
            self.warnings = warnings or []
            self.metadata = metadata or {}
    
    def validate_file_isolated(config):
        """Isolated validation function for subprocess."""
        filename = config['filename']
        errors = []
        warnings = []
        metadata = {}
        
        # Expand path
        expanded_filename = abspath(expanduser(filename))
        
        # File existence check
        if not exists(expanded_filename):
            errors.append(f"File does not exist: {expanded_filename}")
            return ValidationResult(False, filename, errors, warnings, metadata)
        
        # File size validation
        if config['check_file_size']:
            try:
                file_size_bytes = getsize(expanded_filename)
                file_size_mb = file_size_bytes / (1024 * 1024)
                metadata['file_size_mb'] = round(file_size_mb, 2)
                metadata['file_size_bytes'] = file_size_bytes
                
                if file_size_bytes == 0:
                    errors.append("File is empty (0 bytes)")
                elif file_size_mb < config['min_file_size_mb']:
                    errors.append(f"File size ({file_size_mb:.2f} MB) is below minimum threshold ({config['min_file_size_mb']} MB)")
                elif file_size_mb > config['max_file_size_mb']:
                    warnings.append(f"File size ({file_size_mb:.2f} MB) is above typical threshold ({config['max_file_size_mb']} MB)")
            except OSError as e:
                errors.append(f"Unable to determine file size: {e}")
        
        # Filename format validation
        filename_base = basename(expanded_filename)
        metadata['filename'] = filename_base
        
        geos5fp_pattern = re.compile(r'^GEOS\\.fp\\.asm\\.([^.]+)\\.(\\d{8}_\\d{4})\\.V\\d{2}\\.nc4$')
        match = geos5fp_pattern.match(filename_base)
        
        if match:
            product_name = match.group(1)
            time_string = match.group(2)
            metadata['product_name'] = product_name
            metadata['time_string'] = time_string
            
            if config['check_temporal_info']:
                try:
                    parsed_time = datetime.strptime(time_string, "%Y%m%d_%H%M")
                    metadata['parsed_datetime'] = parsed_time.isoformat()
                except ValueError as e:
                    errors.append(f"Invalid timestamp format in filename: {time_string} ({e})")
        else:
            warnings.append(f"Filename does not match expected GEOS-5 FP pattern: {filename_base}")
        
        # NetCDF format validation with robust error handling
        try:
            with rasterio.open(expanded_filename) as dataset:
                metadata['driver'] = dataset.driver
                metadata['count'] = dataset.count
                metadata['width'] = dataset.width
                metadata['height'] = dataset.height
                metadata['crs'] = str(dataset.crs) if dataset.crs else None
                metadata['bounds'] = dataset.bounds
                metadata['transform'] = list(dataset.transform)[:6] if dataset.transform else None
                
                # Spatial reference validation
                if config['check_spatial_ref']:
                    if dataset.crs is None:
                        warnings.append("No coordinate reference system (CRS) information found")
                    else:
                        crs_string = str(dataset.crs).lower()
                        if not ('wgs84' in crs_string or 'epsg:4326' in crs_string or '+proj=longlat' in crs_string):
                            warnings.append(f"Unexpected CRS for global meteorological data: {dataset.crs}")
                    
                    if dataset.bounds:
                        bounds = dataset.bounds
                        if (bounds.left < -180.1 or bounds.right > 180.1 or 
                            bounds.bottom < -90.1 or bounds.top > 90.1):
                            warnings.append(f"Bounds appear to extend beyond valid geographic coordinates: {bounds}")
                        elif not (bounds.right - bounds.left > 350 and bounds.top - bounds.bottom > 170):
                            warnings.append(f"Dataset may not have global coverage: {bounds}")
                
                # Data integrity check
                if config['check_data_integrity'] and dataset.count > 0:
                    try:
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
                    except Exception as e:
                        warnings.append(f"Unable to read sample data for integrity check: {e}")
                
                # Variable validation
                if config['check_variables']:
                    subdatasets = dataset.subdatasets if hasattr(dataset, 'subdatasets') else []
                    if subdatasets:
                        metadata['subdatasets'] = len(subdatasets)
                        variable_names = []
                        for subdataset in subdatasets[:10]:
                            if 'NETCDF:' in subdataset and ':' in subdataset:
                                parts = subdataset.split(':')
                                if len(parts) >= 3:
                                    var_name = parts[-1]
                                    variable_names.append(var_name)
                        metadata['variable_names'] = variable_names[:20]
                        
                        required_variables = config.get('required_variables')
                        if required_variables:
                            missing_vars = [var for var in required_variables if var not in variable_names]
                            if missing_vars:
                                errors.append(f"Missing required variables: {', '.join(missing_vars)}")
                        
                        # Test access to first variable
                        if subdatasets:
                            try:
                                test_var = subdatasets[0]
                                with rasterio.open(test_var) as var_dataset:
                                    metadata['test_variable_width'] = var_dataset.width
                                    metadata['test_variable_height'] = var_dataset.height
                            except Exception as e:
                                warnings.append(f"Unable to access variables as subdatasets: {e}")
                
        except RasterioIOError as e:
            errors.append(f"Unable to open file as NetCDF: {e}")
        except Exception as e:
            errors.append(f"Unexpected error reading NetCDF file: {e}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, filename, errors, warnings, metadata)
    
    # Main subprocess execution
    if __name__ == "__main__":
        if len(sys.argv) != 3:
            sys.exit(1)
        
        config_file = sys.argv[1]
        result_file = sys.argv[2]
        
        try:
            # Load configuration
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Perform validation
            result = validate_file_isolated(config)
            
            # Save result as JSON
            result_dict = {
                'is_valid': result.is_valid,
                'filename': result.filename,
                'errors': result.errors,
                'warnings': result.warnings,
                'metadata': result.metadata
            }
            with open(result_file, 'w') as f:
                json.dump(result_dict, f)
                
        except Exception as e:
            # Create error result as JSON
            error_result = {
                'is_valid': False,
                'filename': config.get('filename', 'unknown'),
                'errors': [f"Subprocess validation exception: {e}"],
                'warnings': [],
                'metadata': {}
            }
            try:
                with open(result_file, 'w') as f:
                    json.dump(error_result, f)
            except:
                pass  # If we can't even write the error, subprocess will handle it
            sys.exit(1)

except ImportError as e:
    # Handle missing dependencies
    error_result = ValidationResult(False, 'unknown', [f"Missing dependencies: {e}"], [], {})
    try:
        with open(sys.argv[2], 'wb') as f:
            pickle.dump(error_result, f)
    except:
        pass
    sys.exit(1)
'''