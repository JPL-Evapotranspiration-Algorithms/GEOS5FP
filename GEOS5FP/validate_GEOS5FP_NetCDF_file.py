"""
GEOS-5 FP NetCDF File Validation Module

This module provides comprehensive validation functionality for GEOS-5 FP NetCDF files
to ensure they are valid, complete, and properly formatted for use with the GEOS5FP package.

Author: Gregory H. Halverson
"""

import logging
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from os.path import exists, expanduser, abspath, basename, getsize
from typing import Dict, List, Optional, Any

import numpy as np

try:
    import netCDF4
except ImportError:
    netCDF4 = None

try:
    import rasterio
    from rasterio.errors import RasterioIOError
except ImportError:
    rasterio = None
    RasterioIOError = Exception

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


def _validate_with_netcdf4(
    expanded_filename: str,
    errors: List[str],
    warnings: List[str],
    metadata: Dict[str, Any],
    check_variables: bool = True,
    check_spatial_ref: bool = True,
    required_variables: Optional[List[str]] = None,
    check_data_integrity: bool = True,
    verbose: bool = False
) -> bool:
    """
    Validate NetCDF4/HDF5 file directly via netCDF4.Dataset without relying on GDAL HDF5 plugins.
    Returns True if netCDF4 opened and validated the file structure; False if netCDF4 is unavailable.
    """
    if netCDF4 is None:
        return False

    try:
        with netCDF4.Dataset(expanded_filename, mode="r") as nc_ds:
            metadata["driver"] = getattr(nc_ds, "file_format", "NETCDF4")
            all_vars = list(nc_ds.variables.keys())
            data_vars = [
                name for name, var in nc_ds.variables.items()
                if len(var.dimensions) >= 2
            ]
            metadata["subdatasets"] = len(data_vars)
            metadata["variable_names"] = data_vars[:20]

            if "lon" in nc_ds.dimensions:
                metadata["width"] = len(nc_ds.dimensions["lon"])
            if "lat" in nc_ds.dimensions:
                metadata["height"] = len(nc_ds.dimensions["lat"])
            metadata["count"] = len(data_vars)

            if check_spatial_ref:
                if "lat" in nc_ds.variables and "lon" in nc_ds.variables:
                    lats = np.asarray(nc_ds.variables["lat"][:], dtype=np.float64)
                    lons = np.asarray(nc_ds.variables["lon"][:], dtype=np.float64)
                    if lats.size > 0 and lons.size > 0:
                        metadata["crs"] = "EPSG:4326"
                        metadata["bounds"] = (
                            float(np.min(lons)),
                            float(np.min(lats)),
                            float(np.max(lons)),
                            float(np.max(lats)),
                        )
                        if (np.min(lons) < -180.5 or np.max(lons) > 360.5 or
                                np.min(lats) < -90.5 or np.max(lats) > 90.5):
                            warnings.append(f"Coordinates appear out of valid geographic bounds: {metadata['bounds']}")
                else:
                    warnings.append("No lat/lon coordinate variables found for spatial reference check")

            if check_variables:
                if required_variables:
                    missing_vars = [var for var in required_variables if var not in all_vars]
                    if missing_vars:
                        errors.append(f"Missing required variables: {', '.join(missing_vars)}")

                if data_vars:
                    test_var = nc_ds.variables[data_vars[0]]
                    if len(test_var.shape) >= 2:
                        metadata["test_variable_height"] = int(test_var.shape[-2])
                        metadata["test_variable_width"] = int(test_var.shape[-1])

            if check_data_integrity and data_vars:
                try:
                    sample_var = nc_ds.variables[data_vars[0]]
                    slices = tuple(slice(0, min(100, dim)) for dim in sample_var.shape)
                    raw_sample = sample_var[slices]
                    if np.ma.isMaskedArray(raw_sample):
                        sample_data = np.ma.filled(raw_sample, np.nan).astype(np.float64)
                    else:
                        sample_data = np.asarray(raw_sample, dtype=np.float64)

                    fill_value = getattr(sample_var, "_FillValue", None)
                    if fill_value is not None and not np.isnan(fill_value):
                        sample_data = np.where(sample_data == fill_value, np.nan, sample_data)

                    total_sample_size = sample_data.size
                    if total_sample_size > 0:
                        valid_data_count = int(np.count_nonzero(~np.isnan(sample_data)))
                        valid_data_ratio = valid_data_count / total_sample_size
                        metadata["sample_valid_data_ratio"] = round(valid_data_ratio, 3)

                        if valid_data_ratio == 0:
                            warnings.append("Sample data contains only NaN values")
                        elif valid_data_ratio < 0.1:
                            warnings.append(f"Sample data has very low valid data ratio: {valid_data_ratio:.3f}")
                except Exception as e:
                    warnings.append(f"Unable to read sample data for integrity check: {e}")

        if verbose:
            logger.info(f"Successfully validated NetCDF file via netCDF4: {expanded_filename}")
        return True

    except Exception as e:
        errors.append(f"Unable to open file as NetCDF: {e}")
        return True


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
    """
    if not filename:
        raise ValueError("filename must be provided")

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
    Direct validation without subprocess isolation.
    """
    errors = []
    warnings = []
    metadata = {}

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

    geos5fp_pattern = re.compile(r'^GEOS\.fp\.asm\.([^.]+)\.(\d{8}_\d{4})\.V\d{2}\.nc4$')
    match = geos5fp_pattern.match(filename_base)

    if match:
        product_name = match.group(1)
        time_string = match.group(2)
        metadata['product_name'] = product_name
        metadata['time_string'] = time_string

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

    # 4. NetCDF format validation (prefer netCDF4 to avoid GDAL HDF5 plugin dependency)
    validated_via_netcdf4 = _validate_with_netcdf4(
        expanded_filename=expanded_filename,
        errors=errors,
        warnings=warnings,
        metadata=metadata,
        check_variables=check_variables,
        check_spatial_ref=check_spatial_ref,
        required_variables=required_variables,
        check_data_integrity=check_data_integrity,
        verbose=verbose,
    )

    if not validated_via_netcdf4:
        # Fallback to rasterio if netCDF4 is not installed
        try:
            with rasterio.open(expanded_filename) as dataset:
                metadata['driver'] = dataset.driver
                metadata['count'] = dataset.count
                metadata['width'] = dataset.width
                metadata['height'] = dataset.height
                metadata['crs'] = str(dataset.crs) if dataset.crs else None
                metadata['bounds'] = dataset.bounds
                metadata['transform'] = list(dataset.transform)[:6] if dataset.transform else None

                if check_spatial_ref:
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

                if check_data_integrity and dataset.count > 0:
                    try:
                        sample_data = dataset.read(
                            1,
                            window=rasterio.windows.Window(0, 0, min(100, dataset.width), min(100, dataset.height))
                        )
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

            if check_variables:
                _validate_netcdf_variables(expanded_filename, errors, warnings, metadata, required_variables, verbose)

        except RasterioIOError as e:
            errors.append(f"Unable to open file as NetCDF: {e}")
        except Exception as e:
            errors.append(f"Unexpected error reading NetCDF file: {e}")

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
    Internal function to validate NetCDF variables using GDAL NetCDF subdatasets (fallback).
    """
    if required_variables is None:
        required_variables = []

    try:
        with rasterio.open(filename) as dataset:
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

                if required_variables:
                    missing_vars = [var for var in required_variables if var not in variable_names]
                    if missing_vars:
                        errors.append(f"Missing required variables: {', '.join(missing_vars)}")

                if subdatasets:
                    try:
                        test_var = subdatasets[0]
                        with rasterio.open(test_var) as var_dataset:
                            metadata['test_variable_width'] = var_dataset.width
                            metadata['test_variable_height'] = var_dataset.height
                    except Exception as e:
                        warnings.append(f"Unable to access variables as subdatasets: {e}")
            else:
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
    """
    import glob

    directory_path = abspath(expanduser(directory))

    if not exists(directory_path):
        raise ValueError(f"Directory does not exist: {directory_path}")

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
            result = validate_GEOS5FP_NetCDF_file(file_path, verbose=False)
            results[basename(file_path)] = result
        except Exception as e:
            results[basename(file_path)] = GEOS5FPValidationResult(
                False, file_path, [f"Validation exception: {e}"], [], {}
            )

    return results


def get_validation_summary(results: Dict[str, GEOS5FPValidationResult]) -> Dict[str, Any]:
    """
    Generate a summary of validation results.
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


def is_valid_GEOS5FP_file(filename: str, **kwargs) -> bool:
    """
    Simple boolean check for GEOS-5 FP file validity.
    """
    try:
        result = validate_GEOS5FP_NetCDF_file(filename, **kwargs)
        return result.is_valid
    except Exception:
        return False


def safe_validate_GEOS5FP_NetCDF_file(filename: str, **kwargs) -> GEOS5FPValidationResult:
    """
    Ultra-safe validation that always uses subprocess isolation.
    """
    kwargs['use_subprocess'] = True
    return validate_GEOS5FP_NetCDF_file(filename, **kwargs)


def quick_validate(filename: str) -> GEOS5FPValidationResult:
    """
    Quick validation with minimal checks for performance.
    """
    return validate_GEOS5FP_NetCDF_file(
        filename,
        check_variables=False,
        check_spatial_ref=False,
        check_temporal_info=True,
        check_data_integrity=False,
        verbose=False,
        use_subprocess=False
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
        validator_script = _create_subprocess_validator_script()
        cmd = [sys.executable, '-c', validator_script, config_file_path, result_file_path]

        if verbose:
            logger.info(f"Running subprocess validation with timeout {timeout_seconds}s")

        try:
            result = subprocess.run(
                cmd,
                timeout=timeout_seconds,
                capture_output=True,
                text=True,
                check=False
            )

            if exists(result_file_path):
                with open(result_file_path, 'r') as f:
                    result_dict = json.load(f)

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
                error_msg = "Subprocess validation failed - no result file created"
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
        for temp_file in [config_file_path, result_file_path]:
            if exists(temp_file):
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass


def _create_subprocess_validator_script() -> str:
    """
    Create the Python script code for subprocess validation.
    """
    return r"""
import sys
import json
import os
from os.path import exists, expanduser, abspath, basename, getsize
import re
from datetime import datetime

os.environ['CPL_LOG'] = '/dev/null'
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'YES'

try:
    import numpy as np
    import netCDF4

    def validate_file_isolated(config):
        filename = config['filename']
        errors = []
        warnings = []
        metadata = {}

        expanded_filename = abspath(expanduser(filename))

        if not exists(expanded_filename):
            errors.append(f"File does not exist: {expanded_filename}")
            return {'is_valid': False, 'filename': filename, 'errors': errors, 'warnings': warnings, 'metadata': metadata}

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

        filename_base = basename(expanded_filename)
        metadata['filename'] = filename_base

        geos5fp_pattern = re.compile(r'^GEOS\.fp\.asm\.([^.]+)\.(\d{8}_\d{4})\.V\d{2}\.nc4$')
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

        try:
            with netCDF4.Dataset(expanded_filename, mode="r") as nc_ds:
                metadata['driver'] = getattr(nc_ds, 'file_format', 'NETCDF4')
                all_vars = list(nc_ds.variables.keys())
                data_vars = [
                    name for name, var in nc_ds.variables.items()
                    if len(var.dimensions) >= 2
                ]
                metadata['subdatasets'] = len(data_vars)
                metadata['variable_names'] = data_vars[:20]

                if 'lon' in nc_ds.dimensions:
                    metadata['width'] = len(nc_ds.dimensions['lon'])
                if 'lat' in nc_ds.dimensions:
                    metadata['height'] = len(nc_ds.dimensions['lat'])
                metadata['count'] = len(data_vars)

                if config['check_spatial_ref']:
                    if 'lat' in nc_ds.variables and 'lon' in nc_ds.variables:
                        lats = np.asarray(nc_ds.variables['lat'][:], dtype=np.float64)
                        lons = np.asarray(nc_ds.variables['lon'][:], dtype=np.float64)
                        if lats.size > 0 and lons.size > 0:
                            metadata['crs'] = 'EPSG:4326'
                            metadata['bounds'] = [
                                float(np.min(lons)),
                                float(np.min(lats)),
                                float(np.max(lons)),
                                float(np.max(lats)),
                            ]
                            if (np.min(lons) < -180.5 or np.max(lons) > 360.5 or
                                    np.min(lats) < -90.5 or np.max(lats) > 90.5):
                                warnings.append(f"Coordinates appear out of valid geographic bounds: {metadata['bounds']}")
                    else:
                        warnings.append("No lat/lon coordinate variables found for spatial reference check")

                if config['check_variables']:
                    required_variables = config.get('required_variables')
                    if required_variables:
                        missing_vars = [var for var in required_variables if var not in all_vars]
                        if missing_vars:
                            errors.append(f"Missing required variables: {', '.join(missing_vars)}")

                    if data_vars:
                        test_var = nc_ds.variables[data_vars[0]]
                        if len(test_var.shape) >= 2:
                            metadata['test_variable_height'] = int(test_var.shape[-2])
                            metadata['test_variable_width'] = int(test_var.shape[-1])

                if config['check_data_integrity'] and data_vars:
                    try:
                        sample_var = nc_ds.variables[data_vars[0]]
                        slices = tuple(slice(0, min(100, dim)) for dim in sample_var.shape)
                        raw_sample = sample_var[slices]
                        if np.ma.isMaskedArray(raw_sample):
                            sample_data = np.ma.filled(raw_sample, np.nan).astype(np.float64)
                        else:
                            sample_data = np.asarray(raw_sample, dtype=np.float64)

                        fill_value = getattr(sample_var, '_FillValue', None)
                        if fill_value is not None and not np.isnan(fill_value):
                            sample_data = np.where(sample_data == fill_value, np.nan, sample_data)

                        total_sample_size = sample_data.size
                        if total_sample_size > 0:
                            valid_data_count = int(np.count_nonzero(~np.isnan(sample_data)))
                            valid_data_ratio = valid_data_count / total_sample_size
                            metadata['sample_valid_data_ratio'] = round(valid_data_ratio, 3)

                            if valid_data_ratio == 0:
                                warnings.append("Sample data contains only NaN values")
                            elif valid_data_ratio < 0.1:
                                warnings.append(f"Sample data has very low valid data ratio: {valid_data_ratio:.3f}")
                    except Exception as e:
                        warnings.append(f"Unable to read sample data for integrity check: {e}")

        except Exception as e:
            errors.append(f"Unable to open file as NetCDF: {e}")

        is_valid = len(errors) == 0
        return {
            'is_valid': is_valid,
            'filename': filename,
            'errors': errors,
            'warnings': warnings,
            'metadata': metadata
        }

    if __name__ == "__main__":
        if len(sys.argv) != 3:
            sys.exit(1)

        config_file = sys.argv[1]
        result_file = sys.argv[2]
        config = {}

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            result_dict = validate_file_isolated(config)
            with open(result_file, 'w') as f:
                json.dump(result_dict, f)

        except Exception as e:
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
            except Exception:
                pass
            sys.exit(1)

except ImportError as e:
    error_result = {
        'is_valid': False,
        'filename': 'unknown',
        'errors': [f"Missing dependencies in validation subprocess: {e}"],
        'warnings': [],
        'metadata': {}
    }
    try:
        if len(sys.argv) >= 3:
            with open(sys.argv[2], 'w') as f:
                json.dump(error_result, f)
    except Exception:
        pass
    sys.exit(1)
"""
