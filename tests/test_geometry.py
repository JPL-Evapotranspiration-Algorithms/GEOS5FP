"""Tests for geometry detection and point extraction."""
import pytest
from shapely.geometry import Point, MultiPoint, Polygon, LineString
from GEOS5FP import GEOS5FPConnection


@pytest.fixture
def conn():
    """Create a connection for testing."""
    return GEOS5FPConnection()


class TestIsPointGeometry:
    """Tests for _is_point_geometry method."""
    
    def test_shapely_point(self, conn):
        """Test that shapely Point is recognized."""
        point = Point(-118.25, 34.05)
        assert conn._is_point_geometry(point) is True
    
    def test_shapely_multipoint(self, conn):
        """Test that shapely MultiPoint is recognized."""
        points = MultiPoint([(-118.25, 34.05), (-122.42, 37.77)])
        assert conn._is_point_geometry(points) is True
    
    def test_polygon_rejected(self, conn):
        """Test that Polygon is not recognized as point."""
        polygon = Polygon([(-118, 34), (-118, 35), (-117, 35), (-117, 34)])
        assert conn._is_point_geometry(polygon) is False
    
    def test_linestring_rejected(self, conn):
        """Test that LineString is not recognized as point."""
        line = LineString([(-118, 34), (-117, 35)])
        assert conn._is_point_geometry(line) is False
    
    def test_none_rejected(self, conn):
        """Test that None is not recognized as point."""
        assert conn._is_point_geometry(None) is False
    
    def test_string_rejected(self, conn):
        """Test that string is not recognized as point."""
        assert conn._is_point_geometry("not a geometry") is False


class TestExtractPoints:
    """Tests for _extract_points method."""
    
    def test_single_point(self, conn):
        """Test extracting coordinates from single Point."""
        point = Point(-118.25, 34.05)
        coords = conn._extract_points(point)
        
        assert isinstance(coords, list)
        assert len(coords) == 1
        assert coords[0] == (34.05, -118.25)  # (lat, lon)
    
    def test_multipoint(self, conn):
        """Test extracting coordinates from MultiPoint."""
        points = MultiPoint([
            (-118.25, 34.05),   # Los Angeles
            (-122.42, 37.77),   # San Francisco
            (-73.94, 40.73)     # New York
        ])
        coords = conn._extract_points(points)
        
        assert isinstance(coords, list)
        assert len(coords) == 3
        assert (34.05, -118.25) in coords
        assert (37.77, -122.42) in coords
        assert (40.73, -73.94) in coords
    
    def test_point_order(self, conn):
        """Test that coordinates are returned as (lat, lon)."""
        lon, lat = -118.25, 34.05
        point = Point(lon, lat)
        coords = conn._extract_points(point)
        
        assert coords[0] == (lat, lon)  # Should be (lat, lon)
    
    def test_empty_multipoint(self, conn):
        """Test handling of empty MultiPoint."""
        points = MultiPoint([])
        coords = conn._extract_points(points)
        
        assert isinstance(coords, list)
        assert len(coords) == 0
    
    def test_extract_from_polygon_raises(self, conn):
        """Test that extracting from non-point geometry raises ValueError."""
        polygon = Polygon([(-118, 34), (-118, 35), (-117, 35), (-117, 34)])
        
        # Should raise ValueError for non-point geometries
        with pytest.raises(ValueError, match="Unsupported geometry type"):
            coords = conn._extract_points(polygon)
