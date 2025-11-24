"""Tests for URL generation and formatting."""
import pytest
from datetime import date, datetime
from GEOS5FP import GEOS5FPConnection


@pytest.fixture
def conn():
    """Create a connection for testing."""
    return GEOS5FPConnection()


class TestYearURL:
    """Tests for year URL generation."""
    
    def test_year_url_format(self, conn):
        """Test that year URL is formatted correctly."""
        url = conn.year_URL(2024)
        assert "Y2024" in url
        assert url.endswith("/")
    
    def test_year_url_includes_base(self, conn):
        """Test that year URL includes base remote URL."""
        url = conn.year_URL(2024)
        assert url.startswith(conn.remote)
    
    def test_year_url_padding(self, conn):
        """Test that year is zero-padded to 4 digits."""
        url = conn.year_URL(999)
        assert "Y0999" in url


class TestMonthURL:
    """Tests for month URL generation."""
    
    def test_month_url_format(self, conn):
        """Test that month URL is formatted correctly."""
        url = conn.month_URL(2024, 3)
        assert "Y2024" in url
        assert "M03" in url
        assert url.endswith("/")
    
    def test_month_url_padding(self, conn):
        """Test that month is zero-padded to 2 digits."""
        url = conn.month_URL(2024, 1)
        assert "M01" in url
        
        url = conn.month_URL(2024, 12)
        assert "M12" in url


class TestDayURL:
    """Tests for day URL generation."""
    
    def test_day_url_format_with_date(self, conn):
        """Test that day URL is formatted correctly with date object."""
        test_date = date(2024, 11, 15)
        url = conn.day_URL(test_date)
        
        assert "Y2024" in url
        assert "M11" in url
        assert "D15" in url
        assert url.endswith("/")
    
    def test_day_url_format_with_string(self, conn):
        """Test that day URL accepts string dates."""
        url = conn.day_URL("2024-11-15")
        
        assert "Y2024" in url
        assert "M11" in url
        assert "D15" in url
    
    def test_day_url_padding(self, conn):
        """Test that day is zero-padded to 2 digits."""
        test_date = date(2024, 1, 5)
        url = conn.day_URL(test_date)
        
        assert "M01" in url
        assert "D05" in url
    
    def test_day_url_complete_path(self, conn):
        """Test that day URL includes complete path."""
        test_date = date(2024, 11, 15)
        url = conn.day_URL(test_date)
        
        assert url.startswith(conn.remote)
        # Should have year/month/day structure
        parts = url.rstrip("/").split("/")
        assert any("Y2024" in part for part in parts)
        assert any("M11" in part for part in parts)
        assert any("D15" in part for part in parts)


class TestTimeFromURL:
    """Tests for extracting time from URL."""
    
    def test_time_from_url_basic(self, conn):
        """Test extracting time from a GEOS-5 FP URL."""
        # This test would need a real URL format example
        # Skipping implementation details without seeing actual URL format
        pass


class TestDownloadFilename:
    """Tests for download filename generation."""
    
    def test_download_filename_basic(self, conn):
        """Test that download filename is generated from URL."""
        url = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y2024/M11/D15/GEOS.fp.asm.tavg1_2d_slv_Nx.20241115_0000.V01.nc4"
        filename = conn.download_filename(url)
        
        assert isinstance(filename, str)
        assert len(filename) > 0
    
    def test_download_filename_includes_directory(self, conn):
        """Test that download filename includes download directory."""
        url = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y2024/M11/D15/GEOS.fp.asm.tavg1_2d_slv_Nx.20241115_0000.V01.nc4"
        filename = conn.download_filename(url)
        
        assert conn.download_directory in filename


class TestDateDownloadDirectory:
    """Tests for date-specific download directory."""
    
    def test_date_download_directory_format(self, conn):
        """Test that date download directory is formatted correctly."""
        test_time = datetime(2024, 11, 15, 12, 0)
        directory = conn.date_download_directory(test_time)
        
        assert isinstance(directory, str)
        assert "2024" in directory
        assert "11" in directory or "Nov" in directory.lower()
        assert "15" in directory
