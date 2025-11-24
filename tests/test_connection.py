"""Tests for GEOS5FPConnection initialization and basic functionality."""
import pytest
from GEOS5FP import GEOS5FPConnection
from pathlib import Path


def test_connection_initialization():
    """Test that GEOS5FPConnection initializes with default parameters."""
    conn = GEOS5FPConnection()
    assert conn is not None
    assert hasattr(conn, 'remote')
    assert hasattr(conn, 'download_directory')


def test_connection_default_remote():
    """Test that default remote URL is set correctly."""
    conn = GEOS5FPConnection()
    assert conn.remote == "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das"


def test_connection_custom_remote():
    """Test that custom remote URL is accepted."""
    custom_url = "https://custom.server.com/data"
    conn = GEOS5FPConnection(remote=custom_url)
    assert conn.remote == custom_url


def test_connection_custom_download_directory(tmp_path):
    """Test that custom download directory is accepted."""
    custom_dir = str(tmp_path / "downloads")
    conn = GEOS5FPConnection(download_directory=custom_dir)
    assert conn.download_directory == custom_dir


def test_connection_repr():
    """Test that connection string representation is valid JSON."""
    import json
    conn = GEOS5FPConnection()
    repr_str = repr(conn)
    
    # Should be valid JSON
    parsed = json.loads(repr_str)
    assert "URL" in parsed
    assert "download_directory" in parsed


def test_connection_default_url_base():
    """Test that DEFAULT_URL_BASE constant is defined."""
    assert hasattr(GEOS5FPConnection, 'DEFAULT_URL_BASE')
    assert isinstance(GEOS5FPConnection.DEFAULT_URL_BASE, str)
    assert GEOS5FPConnection.DEFAULT_URL_BASE.startswith('https://')


def test_connection_default_timeout():
    """Test that default timeout is defined."""
    assert hasattr(GEOS5FPConnection, 'DEFAULT_TIMEOUT_SECONDS')
    assert GEOS5FPConnection.DEFAULT_TIMEOUT_SECONDS > 0


def test_connection_default_retries():
    """Test that default retries is defined."""
    assert hasattr(GEOS5FPConnection, 'DEFAULT_RETRIES')
    assert GEOS5FPConnection.DEFAULT_RETRIES > 0


def test_connection_save_products_default():
    """Test that save_products defaults to False."""
    conn = GEOS5FPConnection()
    assert conn.save_products is False


def test_connection_save_products_true():
    """Test that save_products can be set to True."""
    conn = GEOS5FPConnection(save_products=True)
    assert conn.save_products is True


def test_connection_listings_cache():
    """Test that connection has listings cache."""
    conn = GEOS5FPConnection()
    assert hasattr(conn, '_listings')
    assert isinstance(conn._listings, dict)


def test_connection_filenames_set():
    """Test that connection has filenames set."""
    conn = GEOS5FPConnection()
    assert hasattr(conn, 'filenames')
    assert isinstance(conn.filenames, set)
