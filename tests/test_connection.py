"""Tests for GEOS5FPConnection initialization and basic functionality."""
import json
from unittest.mock import patch
import pytest
from GEOS5FP import GEOS5FPConnection


def test_connection_initialization():
    conn = GEOS5FPConnection()
    assert conn is not None
    assert hasattr(conn, 'remote')
    assert hasattr(conn, 'download_directory')


def test_connection_default_remote():
    conn = GEOS5FPConnection()
    assert conn.remote == "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das"


def test_connection_custom_remote():
    custom_url = "https://custom.server.com/data"
    conn = GEOS5FPConnection(remote=custom_url)
    assert conn.remote == custom_url


def test_connection_custom_download_directory(tmp_path):
    custom_dir = str(tmp_path / "downloads")
    conn = GEOS5FPConnection(download_directory=custom_dir)
    assert conn.download_directory == custom_dir


def test_connection_repr():
    conn = GEOS5FPConnection()
    repr_str = repr(conn)
    parsed = json.loads(repr_str)
    assert "URL" in parsed
    assert "download_directory" in parsed


def test_connection_default_url_base():
    assert hasattr(GEOS5FPConnection, 'DEFAULT_URL_BASE')
    assert isinstance(GEOS5FPConnection.DEFAULT_URL_BASE, str)
    assert GEOS5FPConnection.DEFAULT_URL_BASE.startswith('https://')


def test_connection_default_timeout():
    assert hasattr(GEOS5FPConnection, 'DEFAULT_TIMEOUT_SECONDS')
    assert GEOS5FPConnection.DEFAULT_TIMEOUT_SECONDS > 0


def test_connection_default_retries():
    assert hasattr(GEOS5FPConnection, 'DEFAULT_RETRIES')
    assert GEOS5FPConnection.DEFAULT_RETRIES > 0


def test_connection_save_products_default():
    conn = GEOS5FPConnection()
    assert conn.save_products is False


def test_connection_save_products_deprecated():
    with pytest.warns(DeprecationWarning, match="save_products"):
        conn = GEOS5FPConnection(save_products=True)
    assert conn.save_products is True


def test_connection_listings_cache():
    conn = GEOS5FPConnection()
    assert hasattr(conn, '_listings')
    assert isinstance(conn._listings, dict)


def test_connection_listings_ttl_and_refresh():
    conn = GEOS5FPConnection(listing_ttl_seconds=10)
    calls = []

    def fake_http_listing(url, timeout=None, retries=None):
        calls.append(url)
        return [f"item_{len(calls)}"]

    with patch("GEOS5FP.GEOS5FP_connection.HTTP_listing", side_effect=fake_http_listing):
        first = conn.list_remote_directory("https://example.test/dir/")
        second = conn.list_remote_directory("https://example.test/dir/")
        refreshed = conn.list_remote_directory("https://example.test/dir/", refresh=True)

    assert first == ["item_1"]
    assert second == ["item_1"]
    assert refreshed == ["item_2"]
    assert len(calls) == 2


def test_connection_filenames_set():
    conn = GEOS5FPConnection()
    assert hasattr(conn, 'filenames')
    assert isinstance(conn.filenames, set)
