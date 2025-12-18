"""
Tests for handling pandas Series and array-like time_UTC inputs.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from shapely.geometry import Point, MultiPoint
from GEOS5FP import GEOS5FP


class TestSeriesTimeInput:
    """Test handling of pandas Series time inputs."""
    
    @pytest.fixture
    def connection(self):
        """Create a GEOS5FP connection."""
        return GEOS5FP(download_directory="test_download")
    
    def test_series_with_geometry_array(self, connection):
        """Test pandas Series time_UTC with array of geometries."""
        # Create test data
        times = pd.Series([
            datetime(2019, 1, 28, 21, 8, 10),
            datetime(2019, 1, 28, 22, 40, 54)
        ])
        geometries = [
            Point(-82.2188, 29.7381),
            Point(-106.8425, 32.5907)
        ]
        
        # This should work without raising an error
        result = connection.COT(time_UTC=times, geometry=geometries)
        
        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 'COT' in result.columns
    
    def test_series_with_single_geometry(self, connection):
        """Test pandas Series time_UTC with single geometry."""
        # Create test data
        times = pd.Series([
            datetime(2019, 1, 28, 21, 0, 0),
            datetime(2019, 1, 28, 22, 0, 0)
        ])
        geometry = Point(-82.2188, 29.7381)
        
        # This should work without raising an error
        result = connection.COT(time_UTC=times, geometry=geometry)
        
        # Should return combined results
        assert result is not None
    
    def test_list_time_input(self, connection):
        """Test list of times as input."""
        # Create test data
        times = [
            datetime(2019, 1, 28, 21, 0, 0),
            datetime(2019, 1, 28, 22, 0, 0)
        ]
        geometry = Point(-82.2188, 29.7381)
        
        # This should work without raising an error
        result = connection.COT(time_UTC=times, geometry=geometry)
        
        # Should return combined results
        assert result is not None
    
    def test_tuple_time_input(self, connection):
        """Test tuple of times as input."""
        # Create test data
        times = (
            datetime(2019, 1, 28, 21, 0, 0),
            datetime(2019, 1, 28, 22, 0, 0)
        )
        geometry = Point(-82.2188, 29.7381)
        
        # This should work without raising an error
        result = connection.COT(time_UTC=times, geometry=geometry)
        
        # Should return combined results
        assert result is not None
    
    def test_numpy_array_time_input(self, connection):
        """Test numpy array of times as input."""
        # Create test data
        times = np.array([
            datetime(2019, 1, 28, 21, 0, 0),
            datetime(2019, 1, 28, 22, 0, 0)
        ])
        geometry = Point(-82.2188, 29.7381)
        
        # This should work without raising an error
        result = connection.COT(time_UTC=times, geometry=geometry)
        
        # Should return combined results
        assert result is not None
    
    def test_mismatched_lengths(self, connection):
        """Test error handling for mismatched time and geometry lengths."""
        # Create test data with mismatched lengths
        times = pd.Series([
            datetime(2019, 1, 28, 21, 0, 0),
            datetime(2019, 1, 28, 22, 0, 0)
        ])
        geometries = [
            Point(-82.2188, 29.7381)
            # Only one geometry for two times
        ]
        
        # This should raise a ValueError
        with pytest.raises(ValueError, match="Length mismatch"):
            connection.COT(time_UTC=times, geometry=geometries)
    
    def test_single_time_still_works(self, connection):
        """Test that single time input still works (backward compatibility)."""
        # Single time
        time = datetime(2019, 1, 28, 21, 0, 0)
        geometry = Point(-82.2188, 29.7381)
        
        # This should work without raising an error
        result = connection.COT(time_UTC=time, geometry=geometry)
        
        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
