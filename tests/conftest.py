"""Pytest configuration and shared fixtures."""
import pytest
from datetime import datetime, date
from shapely.geometry import Point, MultiPoint


@pytest.fixture
def sample_point():
    """Create a sample point for testing (Los Angeles)."""
    return Point(-118.25, 34.05)


@pytest.fixture
def sample_multipoint():
    """Create sample multiple points for testing."""
    return MultiPoint([
        (-118.25, 34.05),   # Los Angeles
        (-122.42, 37.77),   # San Francisco
        (-73.94, 40.73)     # New York
    ])


@pytest.fixture
def sample_datetime():
    """Create a sample datetime for testing."""
    return datetime(2024, 11, 15, 12, 0)


@pytest.fixture
def sample_date():
    """Create a sample date for testing."""
    return date(2024, 11, 15)


@pytest.fixture
def sample_time_range():
    """Create a sample time range for testing."""
    from datetime import timedelta
    end = datetime(2024, 11, 15, 0, 0)
    start = end - timedelta(days=7)
    return (start, end)


@pytest.fixture
def known_variables():
    """List of known GEOS-5 FP variables for testing."""
    return [
        "SM", "SFMC", "LAI", "LHLAND", "EFLUX",
        "Ta", "Ta_K", "Ts", "Ts_K", "Tmin", "Tmin_K",
        "PS", "Q", "U2M", "V2M",
        "vapor_kgsqm", "vapor_gccm",
        "ozone_dobson", "ozone_cm",
        "SWin", "SWTDN",
        "ALBVISDR", "ALBVISDF", "ALBNIRDF", "ALBNIRDR", "ALBEDO",
        "AOT", "COT", "PARDR", "PARDF", "CO2SC"
    ]


# Markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests without network calls"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that may require network"
    )
    config.addinivalue_line(
        "markers", "slow: Slow-running tests"
    )
    config.addinivalue_line(
        "markers", "network: Tests requiring internet connection"
    )
