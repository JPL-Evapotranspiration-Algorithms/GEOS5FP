"""Tests for custom exceptions."""
import pytest
from GEOS5FP.exceptions import (
    GEOS5FPDayNotAvailable,
    GEOS5FPGranuleNotAvailable,
    FailedGEOS5FPDownload
)


class TestExceptionImports:
    """Test that all exceptions can be imported."""
    
    def test_import_day_not_available(self):
        """Test that GEOS5FPDayNotAvailable can be imported."""
        assert GEOS5FPDayNotAvailable is not None
    
    def test_import_granule_not_available(self):
        """Test that GEOS5FPGranuleNotAvailable can be imported."""
        assert GEOS5FPGranuleNotAvailable is not None
    
    def test_import_failed_download(self):
        """Test that FailedGEOS5FPDownload can be imported."""
        assert FailedGEOS5FPDownload is not None


class TestDayNotAvailableException:
    """Tests for GEOS5FPDayNotAvailable exception."""
    
    def test_raise_day_not_available(self):
        """Test raising GEOS5FPDayNotAvailable."""
        with pytest.raises(GEOS5FPDayNotAvailable):
            raise GEOS5FPDayNotAvailable("Test message")
    
    def test_day_not_available_message(self):
        """Test that exception message is preserved."""
        message = "Day 2024-11-15 not available"
        with pytest.raises(GEOS5FPDayNotAvailable) as exc_info:
            raise GEOS5FPDayNotAvailable(message)
        
        assert message in str(exc_info.value)
    
    def test_day_not_available_is_exception(self):
        """Test that GEOS5FPDayNotAvailable is an Exception."""
        assert issubclass(GEOS5FPDayNotAvailable, Exception)


class TestGranuleNotAvailableException:
    """Tests for GEOS5FPGranuleNotAvailable exception."""
    
    def test_raise_granule_not_available(self):
        """Test raising GEOS5FPGranuleNotAvailable."""
        with pytest.raises(GEOS5FPGranuleNotAvailable):
            raise GEOS5FPGranuleNotAvailable("Test message")
    
    def test_granule_not_available_message(self):
        """Test that exception message is preserved."""
        message = "Granule not found at URL"
        with pytest.raises(GEOS5FPGranuleNotAvailable) as exc_info:
            raise GEOS5FPGranuleNotAvailable(message)
        
        assert message in str(exc_info.value)
    
    def test_granule_not_available_is_exception(self):
        """Test that GEOS5FPGranuleNotAvailable is an Exception."""
        assert issubclass(GEOS5FPGranuleNotAvailable, Exception)


class TestFailedDownloadException:
    """Tests for FailedGEOS5FPDownload exception."""
    
    def test_raise_failed_download(self):
        """Test raising FailedGEOS5FPDownload."""
        with pytest.raises(FailedGEOS5FPDownload):
            raise FailedGEOS5FPDownload("Test message")
    
    def test_failed_download_message(self):
        """Test that exception message is preserved."""
        message = "Download failed for URL"
        with pytest.raises(FailedGEOS5FPDownload) as exc_info:
            raise FailedGEOS5FPDownload(message)
        
        assert message in str(exc_info.value)
    
    def test_failed_download_is_exception(self):
        """Test that FailedGEOS5FPDownload is an Exception."""
        assert issubclass(FailedGEOS5FPDownload, Exception)


class TestExceptionCatching:
    """Tests for catching exceptions."""
    
    def test_catch_day_not_available_as_exception(self):
        """Test that GEOS5FPDayNotAvailable can be caught as Exception."""
        try:
            raise GEOS5FPDayNotAvailable("Test")
        except Exception as e:
            assert isinstance(e, GEOS5FPDayNotAvailable)
    
    def test_catch_granule_not_available_as_exception(self):
        """Test that GEOS5FPGranuleNotAvailable can be caught as Exception."""
        try:
            raise GEOS5FPGranuleNotAvailable("Test")
        except Exception as e:
            assert isinstance(e, GEOS5FPGranuleNotAvailable)
    
    def test_catch_failed_download_as_exception(self):
        """Test that FailedGEOS5FPDownload can be caught as Exception."""
        try:
            raise FailedGEOS5FPDownload("Test")
        except Exception as e:
            assert isinstance(e, FailedGEOS5FPDownload)


class TestExceptionHierarchy:
    """Tests for exception inheritance."""
    
    def test_exceptions_inherit_from_exception(self):
        """Test that all custom exceptions inherit from Exception."""
        exceptions = [
            GEOS5FPDayNotAvailable,
            GEOS5FPGranuleNotAvailable,
            FailedGEOS5FPDownload
        ]
        
        for exc_class in exceptions:
            assert issubclass(exc_class, Exception)
