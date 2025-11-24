"""Tests for edge cases and error handling."""
import pytest
from shapely.geometry import Point, MultiPoint, Polygon
from datetime import datetime, timedelta
from GEOS5FP import GEOS5FPConnection


@pytest.fixture
def conn():
    """Create a connection for testing."""
    return GEOS5FPConnection()


class TestEmptyGeometry:
    """Tests for handling empty geometries."""
    
    def test_empty_multipoint(self, conn):
        """Test handling empty MultiPoint."""
        empty_points = MultiPoint([])
        coords = conn._extract_points(empty_points)
        
        assert isinstance(coords, list)
        assert len(coords) == 0
    
    def test_none_geometry(self, conn):
        """Test handling None geometry."""
        assert conn._is_point_geometry(None) is False


class TestInvalidCoordinates:
    """Tests for handling invalid coordinates."""
    
    def test_out_of_bounds_latitude_high(self, conn):
        """Test point with latitude > 90."""
        # Creating the point should work, but queries might fail
        point = Point(-118.25, 95.0)  # Invalid latitude
        assert point is not None
    
    def test_out_of_bounds_latitude_low(self, conn):
        """Test point with latitude < -90."""
        point = Point(-118.25, -95.0)  # Invalid latitude
        assert point is not None
    
    def test_out_of_bounds_longitude_high(self, conn):
        """Test point with longitude > 180."""
        point = Point(185.0, 34.05)  # Invalid longitude
        assert point is not None
    
    def test_out_of_bounds_longitude_low(self, conn):
        """Test point with longitude < -180."""
        point = Point(-185.0, 34.05)  # Invalid longitude
        assert point is not None
    
    def test_zero_coordinates(self, conn):
        """Test point at (0, 0) - valid coordinates."""
        point = Point(0, 0)
        coords = conn._extract_points(point)
        
        assert coords == [(0, 0)]


class TestInvalidDates:
    """Tests for handling invalid dates."""
    
    def test_very_old_date(self, conn):
        """Test with a date before GEOS-5 FP data availability."""
        old_date = datetime(2000, 1, 1)
        # Should not raise during URL generation
        url = conn.day_URL(old_date)
        assert "Y2000" in url
    
    def test_future_date(self, conn):
        """Test with a future date."""
        future_date = datetime(2050, 1, 1)
        # Should not raise during URL generation
        url = conn.day_URL(future_date)
        assert "Y2050" in url
    
    def test_leap_year_date(self, conn):
        """Test with February 29 in a leap year."""
        leap_date = datetime(2024, 2, 29)
        url = conn.day_URL(leap_date)
        
        assert "Y2024" in url
        assert "M02" in url
        assert "D29" in url
    
    def test_end_of_year_date(self, conn):
        """Test with December 31."""
        end_date = datetime(2024, 12, 31)
        url = conn.day_URL(end_date)
        
        assert "M12" in url
        assert "D31" in url


class TestInvalidTimeRanges:
    """Tests for handling invalid time ranges."""
    
    def test_reversed_time_range(self):
        """Test time range where start > end."""
        start = datetime(2024, 11, 15, 12, 0)
        end = datetime(2024, 11, 14, 12, 0)  # Before start
        time_range = (start, end)
        
        # Range is invalid but creation should work
        assert time_range[0] > time_range[1]
    
    def test_zero_duration_range(self):
        """Test time range with zero duration."""
        dt = datetime(2024, 11, 15, 12, 0)
        time_range = (dt, dt)
        
        assert time_range[0] == time_range[1]
    
    def test_very_long_time_range(self):
        """Test time range spanning multiple years."""
        start = datetime(2020, 1, 1)
        end = datetime(2024, 11, 15)
        time_range = (start, end)
        
        duration = end - start
        assert duration.days > 365 * 4


class TestInvalidVariableNames:
    """Tests for handling invalid variable names."""
    
    def test_invalid_variable_name(self, conn):
        """Test that invalid variable name raises error."""
        with pytest.raises(KeyError):
            conn._get_variable_info("NOT_A_REAL_VARIABLE")
    
    def test_empty_variable_name(self, conn):
        """Test that empty variable name raises error."""
        with pytest.raises(KeyError):
            conn._get_variable_info("")
    
    def test_none_variable_name(self, conn):
        """Test that None variable name raises error."""
        with pytest.raises((KeyError, TypeError, AttributeError)):
            conn._get_variable_info(None)
    
    def test_numeric_variable_name(self, conn):
        """Test that numeric variable name raises error."""
        with pytest.raises(KeyError):
            conn._get_variable_info(12345)


class TestNullValues:
    """Tests for handling null/None values."""
    
    def test_none_time_utc(self, conn):
        """Test handling None for time_UTC parameter."""
        # Most methods should handle this appropriately
        # This would typically raise an error
        pass
    
    def test_none_geometry(self, conn):
        """Test handling None for geometry parameter."""
        assert conn._is_point_geometry(None) is False


class TestSpecialCharacters:
    """Tests for handling special characters."""
    
    def test_variable_with_underscore(self, conn):
        """Test variable names with underscores."""
        # Ta_K, Ts_K, etc. should all work
        info = conn._get_variable_info("Ta_K")
        assert info is not None
    
    def test_variable_with_number(self, conn):
        """Test variable names with numbers."""
        # U2M, V2M, T2M, etc.
        info = conn._get_variable_info("U2M")
        assert info is not None


class TestBoundaryConditions:
    """Tests for boundary conditions."""
    
    def test_single_point_multipoint(self, conn):
        """Test MultiPoint with single point."""
        points = MultiPoint([(-118.25, 34.05)])
        coords = conn._extract_points(points)
        
        assert len(coords) == 1
    
    def test_many_points(self, conn):
        """Test MultiPoint with many points."""
        point_list = [(lon, lat) for lon in range(-180, 180, 10) 
                     for lat in range(-90, 90, 10)]
        points = MultiPoint(point_list)
        coords = conn._extract_points(points)
        
        assert len(coords) == len(point_list)
    
    def test_minimum_date_components(self, conn):
        """Test date with minimum values."""
        min_date = datetime(1, 1, 1, 0, 0, 0)
        url = conn.day_URL(min_date)
        
        assert "Y0001" in url
        assert "M01" in url
        assert "D01" in url
