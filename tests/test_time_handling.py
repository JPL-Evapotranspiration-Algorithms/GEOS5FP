"""Tests for time and date handling."""
import pytest
from datetime import date, datetime, time, timedelta, timezone
from dateutil import parser
from GEOS5FP import GEOS5FPConnection


@pytest.fixture
def conn():
    """Create a connection for testing."""
    return GEOS5FPConnection()


class TestDateTimeParsing:
    """Tests for date/time parsing."""
    
    def test_parse_iso_datetime_string(self):
        """Test parsing ISO format datetime string."""
        dt_str = "2024-11-15T12:00:00"
        dt = parser.parse(dt_str)
        
        assert isinstance(dt, datetime)
        assert dt.year == 2024
        assert dt.month == 11
        assert dt.day == 15
        assert dt.hour == 12
    
    def test_parse_simple_date_string(self):
        """Test parsing simple date string."""
        date_str = "2024-11-15"
        dt = parser.parse(date_str)
        
        assert isinstance(dt, datetime)
        assert dt.year == 2024
        assert dt.month == 11
        assert dt.day == 15
    
    def test_parse_date_with_time(self):
        """Test parsing date with time."""
        dt_str = "2024-11-15 12:30:45"
        dt = parser.parse(dt_str)
        
        assert dt.year == 2024
        assert dt.month == 11
        assert dt.day == 15
        assert dt.hour == 12
        assert dt.minute == 30
        assert dt.second == 45


class TestTimeRanges:
    """Tests for time range handling."""
    
    def test_create_time_range(self):
        """Test creating a time range tuple."""
        end_time = datetime(2024, 11, 15, 0, 0)
        start_time = end_time - timedelta(days=7)
        time_range = (start_time, end_time)
        
        assert isinstance(time_range, tuple)
        assert len(time_range) == 2
        assert time_range[0] < time_range[1]
    
    def test_time_range_duration(self):
        """Test calculating time range duration."""
        end_time = datetime(2024, 11, 15, 0, 0)
        start_time = end_time - timedelta(days=7)
        duration = end_time - start_time
        
        assert duration.days == 7
    
    def test_time_range_hours(self):
        """Test time range in hours."""
        end_time = datetime(2024, 11, 15, 12, 0)
        start_time = end_time - timedelta(hours=6)
        time_range = (start_time, end_time)
        
        duration = time_range[1] - time_range[0]
        assert duration.total_seconds() == 6 * 3600


class TestDateConversions:
    """Tests for date conversions."""
    
    def test_datetime_to_date(self):
        """Test converting datetime to date."""
        dt = datetime(2024, 11, 15, 12, 30)
        d = dt.date()
        
        assert isinstance(d, date)
        assert d.year == 2024
        assert d.month == 11
        assert d.day == 15
    
    def test_date_to_string(self):
        """Test converting date to string."""
        d = date(2024, 11, 15)
        date_str = d.strftime("%Y-%m-%d")
        
        assert date_str == "2024-11-15"
    
    def test_datetime_to_string(self):
        """Test converting datetime to string."""
        dt = datetime(2024, 11, 15, 12, 30, 45)
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        
        assert dt_str == "2024-11-15 12:30:45"


class TestDateComparisons:
    """Tests for date comparisons."""
    
    def test_datetime_comparison(self):
        """Test comparing datetime objects."""
        dt1 = datetime(2024, 11, 15, 12, 0)
        dt2 = datetime(2024, 11, 15, 13, 0)
        
        assert dt1 < dt2
        assert dt2 > dt1
        assert dt1 != dt2
    
    def test_date_comparison(self):
        """Test comparing date objects."""
        d1 = date(2024, 11, 15)
        d2 = date(2024, 11, 16)
        
        assert d1 < d2
        assert d2 > d1
        assert d1 != d2
    
    def test_date_equality(self):
        """Test date equality."""
        d1 = date(2024, 11, 15)
        d2 = date(2024, 11, 15)
        
        assert d1 == d2


class TestTimeDelta:
    """Tests for time delta operations."""
    
    def test_add_days(self):
        """Test adding days to a date."""
        d = date(2024, 11, 15)
        new_d = d + timedelta(days=7)
        
        assert new_d.day == 22
    
    def test_subtract_days(self):
        """Test subtracting days from a date."""
        d = date(2024, 11, 15)
        new_d = d - timedelta(days=7)
        
        assert new_d.day == 8
    
    def test_add_hours(self):
        """Test adding hours to a datetime."""
        dt = datetime(2024, 11, 15, 12, 0)
        new_dt = dt + timedelta(hours=6)
        
        assert new_dt.hour == 18
    
    def test_difference_between_dates(self):
        """Test calculating difference between dates."""
        d1 = date(2024, 11, 15)
        d2 = date(2024, 11, 22)
        diff = d2 - d1
        
        assert diff.days == 7


class TestUTCHandling:
    """Tests for UTC time handling."""
    
    def test_utcnow(self):
        """Test getting current UTC time."""
        now = datetime.now(timezone.utc)
        
        assert isinstance(now, datetime)
        assert now.year >= 2024
    
    def test_datetime_without_timezone(self):
        """Test that datetime is created without timezone."""
        dt = datetime(2024, 11, 15, 12, 0)
        
        assert dt.tzinfo is None
