"""
Unit tests for GEOS-5 FP NetCDF file validation functionality.
"""

import json
import os
import subprocess
import sys
import tempfile
from unittest.mock import patch

import netCDF4
import numpy as np
import pytest
from rasterio.errors import RasterioIOError

from GEOS5FP import (
    validate_GEOS5FP_NetCDF_file,
    validate_GEOS5FP_directory,
    is_valid_GEOS5FP_file,
    quick_validate,
    GEOS5FPValidationError,
    GEOS5FPValidationResult,
    get_validation_summary
)
from GEOS5FP.validate_GEOS5FP_NetCDF_file import _create_subprocess_validator_script


def _create_valid_nc4(path: str) -> None:
    with netCDF4.Dataset(path, mode="w", format="NETCDF4") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("lat", 10)
        ds.createDimension("lon", 20)
        lats = ds.createVariable("lat", "f8", ("lat",))
        lons = ds.createVariable("lon", "f8", ("lon",))
        lats[:] = np.linspace(-90, 90, 10)
        lons[:] = np.linspace(-180, 180, 20)
        var = ds.createVariable("TOTEXTTAU", "f4", ("time", "lat", "lon"), fill_value=1e15)
        var[:] = np.random.rand(1, 10, 20).astype(np.float32)


class TestGEOS5FPValidation:
    """Test class for GEOS-5 FP validation functions."""

    def test_validation_result_creation(self):
        result = GEOS5FPValidationResult(
            is_valid=True,
            filename="test.nc4",
            errors=[],
            warnings=["test warning"],
            metadata={"size": 100}
        )
        assert result.is_valid is True
        assert result.filename == "test.nc4"
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
        assert result.metadata["size"] == 100
        assert bool(result) is True

    def test_validation_result_string_representation(self):
        result = GEOS5FPValidationResult(
            is_valid=False,
            filename="test.nc4",
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
            metadata={"size": 100}
        )
        str_repr = str(result)
        assert "INVALID" in str_repr
        assert "test.nc4" in str_repr
        assert "Error 1" in str_repr
        assert "Warning 1" in str_repr
        assert "size: 100" in str_repr

    def test_nonexistent_file(self):
        result = validate_GEOS5FP_NetCDF_file("nonexistent_file.nc4")
        assert result.is_valid is False
        assert any("does not exist" in error for error in result.errors)

    def test_empty_filename(self):
        with pytest.raises(ValueError, match="filename must be provided"):
            validate_GEOS5FP_NetCDF_file("")

    def test_none_filename(self):
        with pytest.raises(ValueError, match="filename must be provided"):
            validate_GEOS5FP_NetCDF_file(None)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".nc4", delete=False) as tmp:
            empty_file = tmp.name

        try:
            result = validate_GEOS5FP_NetCDF_file(empty_file)
            assert result.is_valid is False
            assert any("empty" in error.lower() for error in result.errors)
        finally:
            os.unlink(empty_file)

    def test_netcdf4_validation_without_gdal_hdf5_plugin(self, tmp_path):
        proper_path = tmp_path / "GEOS.fp.asm.tavg3_2d_aer_Nx.20250214_1200.V01.nc4"
        _create_valid_nc4(str(proper_path))

        # Even if rasterio/GDAL is broken or missing gdal_HDF5.so, netCDF4 validation succeeds
        with patch("rasterio.open", side_effect=RasterioIOError("plugin gdal_HDF5.so is not available")):
            result_direct = validate_GEOS5FP_NetCDF_file(
                str(proper_path), min_file_size_mb=0.001, use_subprocess=False
            )
            assert result_direct.is_valid is True
            assert "TOTEXTTAU" in result_direct.metadata["variable_names"]

        result_subproc = validate_GEOS5FP_NetCDF_file(
            str(proper_path), min_file_size_mb=0.001, use_subprocess=True
        )
        assert result_subproc.is_valid is True
        assert result_subproc.metadata["product_name"] == "tavg3_2d_aer_Nx"

    def test_subprocess_importerror_serializes_clean_json(self, tmp_path):
        script = _create_subprocess_validator_script()
        # Force ImportError inside the subprocess script
        broken_script = script.replace("import netCDF4", "raise ImportError('simulated missing netCDF4')")
        config_path = tmp_path / "config.json"
        result_path = tmp_path / "result.json"
        config_path.write_text(json.dumps({"filename": "dummy.nc4"}))

        proc = subprocess.run(
            [sys.executable, "-c", broken_script, str(config_path), str(result_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 1
        assert result_path.exists()
        payload = json.loads(result_path.read_text())
        assert payload["is_valid"] is False
        assert "Missing dependencies" in payload["errors"][0]

    def test_quick_validate(self):
        result = quick_validate("nonexistent_file.nc4")
        assert isinstance(result, GEOS5FPValidationResult)
        assert result.is_valid is False

    def test_is_valid_convenience_function(self):
        is_valid = is_valid_GEOS5FP_file("nonexistent_file.nc4")
        assert is_valid is False
        assert isinstance(is_valid, bool)

    def test_directory_validation_nonexistent(self):
        with pytest.raises(ValueError, match="Directory does not exist"):
            validate_GEOS5FP_directory("/nonexistent/directory")

    def test_directory_validation_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results = validate_GEOS5FP_directory(temp_dir)
            assert isinstance(results, dict)
            assert len(results) == 0

    def test_validation_summary(self):
        results = {
            "file1.nc4": GEOS5FPValidationResult(True, "file1.nc4", [], ["warning"], {}),
            "file2.nc4": GEOS5FPValidationResult(False, "file2.nc4", ["error"], [], {}),
            "file3.nc4": GEOS5FPValidationResult(True, "file3.nc4", [], [], {})
        }
        summary = get_validation_summary(results)
        assert summary["total_files"] == 3
        assert summary["valid_files"] == 2
        assert summary["invalid_files"] == 1
        assert summary["validation_rate"] == 66.7
        assert summary["total_errors"] == 1
        assert summary["total_warnings"] == 1
