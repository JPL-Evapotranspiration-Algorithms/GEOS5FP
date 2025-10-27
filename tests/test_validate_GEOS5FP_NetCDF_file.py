"""
Unit tests for GEOS-5 FP NetCDF file validation functionality.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from GEOS5FP import (
    validate_GEOS5FP_NetCDF_file,
    validate_GEOS5FP_directory,
    is_valid_GEOS5FP_file,
    quick_validate,
    GEOS5FPValidationError,
    GEOS5FPValidationResult,
    get_validation_summary
)


class TestGEOS5FPValidation:
    """Test class for GEOS-5 FP validation functions."""
    
    def test_validation_result_creation(self):
        """Test GEOS5FPValidationResult creation and properties."""
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
        """Test string representation of validation results."""
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
        """Test validation of non-existent file."""
        result = validate_GEOS5FP_NetCDF_file("nonexistent_file.nc4")
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("does not exist" in error for error in result.errors)
    
    def test_empty_filename(self):
        """Test validation with empty filename."""
        with pytest.raises(ValueError, match="filename must be provided"):
            validate_GEOS5FP_NetCDF_file("")
    
    def test_none_filename(self):
        """Test validation with None filename."""
        with pytest.raises(ValueError, match="filename must be provided"):
            validate_GEOS5FP_NetCDF_file(None)
    
    def test_empty_file(self):
        """Test validation of empty file."""
        with tempfile.NamedTemporaryFile(suffix=".nc4", delete=False) as tmp:
            empty_file = tmp.name
        
        try:
            result = validate_GEOS5FP_NetCDF_file(empty_file)
            assert result.is_valid is False
            assert any("empty" in error.lower() for error in result.errors)
        finally:
            os.unlink(empty_file)
    
    def test_filename_pattern_validation(self):
        """Test GEOS-5 FP filename pattern validation."""
        with tempfile.NamedTemporaryFile(suffix=".nc4", delete=False) as tmp:
            tmp.write(b"some content")
            temp_file = tmp.name
        
        # Test with invalid filename
        try:
            result = validate_GEOS5FP_NetCDF_file(temp_file)
            # Should have warnings about filename pattern but not errors (unless other issues)
            # The file won't be readable as NetCDF, so there will be errors
            assert len(result.warnings) > 0 or len(result.errors) > 0
        finally:
            os.unlink(temp_file)
        
        # Test proper filename format (but fake content)
        proper_name = "GEOS.fp.asm.tavg1_2d_slv_Nx.20250214_1200.V01.nc4"
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"fake content")
            fake_file = tmp.name
        
        proper_path = os.path.join(os.path.dirname(fake_file), proper_name)
        os.rename(fake_file, proper_path)
        
        try:
            result = validate_GEOS5FP_NetCDF_file(proper_path)
            # Should parse metadata correctly
            assert "product_name" in result.metadata
            assert "time_string" in result.metadata
            assert result.metadata["product_name"] == "tavg1_2d_slv_Nx"
            assert result.metadata["time_string"] == "20250214_1200"
        finally:
            if os.path.exists(proper_path):
                os.unlink(proper_path)
    
    def test_quick_validate(self):
        """Test quick validation function."""
        result = quick_validate("nonexistent_file.nc4")
        assert isinstance(result, GEOS5FPValidationResult)
        assert result.is_valid is False
    
    def test_is_valid_convenience_function(self):
        """Test is_valid_GEOS5FP_file convenience function."""
        is_valid = is_valid_GEOS5FP_file("nonexistent_file.nc4")
        assert is_valid is False
        assert isinstance(is_valid, bool)
    
    def test_directory_validation_nonexistent(self):
        """Test directory validation with non-existent directory."""
        with pytest.raises(ValueError, match="Directory does not exist"):
            validate_GEOS5FP_directory("/nonexistent/directory")
    
    def test_directory_validation_empty(self):
        """Test directory validation with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = validate_GEOS5FP_directory(temp_dir)
            assert isinstance(results, dict)
            assert len(results) == 0
    
    def test_directory_validation_with_files(self):
        """Test directory validation with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            file1 = os.path.join(temp_dir, "GEOS.fp.asm.tavg1_2d_slv_Nx.20250214_1200.V01.nc4")
            file2 = os.path.join(temp_dir, "invalid_file.txt")
            
            with open(file1, 'wb') as f:
                f.write(b"fake content")
            with open(file2, 'w') as f:
                f.write("not netcdf")
            
            results = validate_GEOS5FP_directory(temp_dir, pattern="*.nc4")
            assert len(results) == 1
            assert "GEOS.fp.asm.tavg1_2d_slv_Nx.20250214_1200.V01.nc4" in results
    
    def test_validation_summary(self):
        """Test validation summary generation."""
        # Create mock results
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
    
    @patch('rasterio.open')
    def test_successful_netcdf_validation(self, mock_rasterio_open):
        """Test successful NetCDF validation with mocked rasterio."""
        from rasterio.coords import BoundingBox
        import numpy as np
        
        # Create proper bounding box mock
        bounds_mock = BoundingBox(left=-180, bottom=-90, right=180, top=90)
        
        # Create a mock dataset
        mock_dataset = MagicMock()
        mock_dataset.driver = "NetCDF"
        mock_dataset.count = 1
        mock_dataset.width = 360
        mock_dataset.height = 180
        mock_dataset.crs = "EPSG:4326"
        mock_dataset.bounds = bounds_mock
        mock_dataset.transform = [1, 0, -180, 0, -1, 90]
        mock_dataset.subdatasets = ["NETCDF:file.nc4:T2M", "NETCDF:file.nc4:QV2M"]
        
        # Mock the read method for data integrity check
        sample_data = np.random.rand(100, 100)  # Valid sample data
        mock_dataset.read.return_value = sample_data
        
        # Mock the rasterio.open function
        mock_rasterio_open.return_value.__enter__ = MagicMock(return_value=mock_dataset)
        mock_rasterio_open.return_value.__exit__ = MagicMock(return_value=False)
        
        # Create a temporary file with proper naming and sufficient size (>0.1 MB)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"fake netcdf content" * 10000)  # Make it large enough to pass size check
            temp_file = tmp.name
        
        proper_name = "GEOS.fp.asm.tavg1_2d_slv_Nx.20250214_1200.V01.nc4"
        proper_path = os.path.join(os.path.dirname(temp_file), proper_name)
        os.rename(temp_file, proper_path)
        
        try:
            # Use direct validation for mocked test (subprocess can't see mocks)
            result = validate_GEOS5FP_NetCDF_file(proper_path, min_file_size_mb=0.01, use_subprocess=False)
            
            # Should be valid with mocked successful rasterio
            assert result.is_valid is True
            assert result.metadata["driver"] == "NetCDF"
            assert result.metadata["width"] == 360
            assert result.metadata["height"] == 180
            
        finally:
            if os.path.exists(proper_path):
                os.unlink(proper_path)
    
    def test_file_size_validation(self):
        """Test file size validation."""
        with tempfile.NamedTemporaryFile(suffix=".nc4", delete=False) as tmp:
            # Write content to make file larger than minimum
            tmp.write(b"x" * 1024 * 200)  # 200 KB
            temp_file = tmp.name
        
        try:
            result = validate_GEOS5FP_NetCDF_file(temp_file, min_file_size_mb=0.1)
            # Should pass size check
            assert result.metadata["file_size_mb"] > 0.1
            
            # Test with size too large
            result_large = validate_GEOS5FP_NetCDF_file(temp_file, max_file_size_mb=0.1)
            assert any("above typical threshold" in warning for warning in result_large.warnings)
            
        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__])