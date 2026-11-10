import builtins
import errno
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests


connection_module = importlib.import_module("GEOS5FP.GEOS5FP_connection")
validation_module = importlib.import_module("GEOS5FP.validate_GEOS5FP_NetCDF_file")


class ChunkedResponse:
    headers = {"content-length": "6"}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        assert chunk_size == 1024 * 1024
        yield b"GEOS"
        yield b"FP"


def test_download_uses_pooch_staging_and_retries(tmp_path, monkeypatch):
    download_directory = tmp_path / "synced"
    destination = download_directory / "granule.nc4"
    staging_root = tmp_path / "cache"
    monkeypatch.setattr(
        connection_module.pooch,
        "os_cache",
        lambda app_name: str(staging_root),
    )

    session = MagicMock()
    session.__enter__.return_value = session
    session.head.return_value = SimpleNamespace(
        status_code=200,
        raise_for_status=lambda: None,
    )
    monkeypatch.setattr(
        connection_module,
        "create_robust_session",
        lambda: session,
    )

    download_attempts = []

    def get(url, **kwargs):
        download_attempts.append(kwargs)
        if len(download_attempts) == 1:
            raise requests.ConnectionError("temporary network failure")
        return ChunkedResponse()

    monkeypatch.setattr(connection_module.requests, "get", get)

    opened_paths = []
    real_open = builtins.open

    def record_open(path, mode="r", *args, **kwargs):
        if mode == "w+b":
            opened_paths.append(Path(path))
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", record_open)
    real_replace = connection_module.os.replace
    replace_attempts = []

    def replace_across_devices_once(source, destination):
        replace_attempts.append((source, destination))
        if len(replace_attempts) == 1:
            raise OSError(errno.EXDEV, "simulated cross-device move")
        return real_replace(source, destination)

    monkeypatch.setattr(connection_module.os, "replace", replace_across_devices_once)
    monkeypatch.setattr(
        validation_module,
        "validate_GEOS5FP_NetCDF_file",
        lambda filename, verbose: SimpleNamespace(
            is_valid=True,
            metadata={},
            errors=[],
        ),
    )

    connection = connection_module.GEOS5FPConnection(
        download_directory=str(download_directory),
    )
    result = connection.download_file(
        URL="https://example.test/granule.nc4",
        filename=str(destination),
        retries=2,
        wait_seconds=0,
    )

    assert result.filename == str(destination)
    assert destination.read_bytes() == b"GEOSFP"
    assert session.head.call_args.kwargs["verify"] is True
    assert len(download_attempts) == 2
    assert all(call["stream"] is True for call in download_attempts)
    assert all(call["timeout"] == 120 for call in download_attempts)
    assert all(call.get("verify", True) is True for call in download_attempts)
    assert len(opened_paths) == 2
    assert all(path.is_relative_to(staging_root) for path in opened_paths)
    assert all(not path.is_relative_to(download_directory) for path in opened_paths)
    assert all(not path.exists() for path in opened_paths)
    assert len(replace_attempts) == 2


def test_download_maps_missing_day(tmp_path, monkeypatch):
    session = MagicMock()
    session.__enter__.return_value = session
    session.head.side_effect = [
        SimpleNamespace(status_code=404),
        SimpleNamespace(status_code=404),
    ]
    monkeypatch.setattr(
        connection_module,
        "create_robust_session",
        lambda: session,
    )
    connection = connection_module.GEOS5FPConnection(
        download_directory=str(tmp_path),
    )

    with pytest.raises(connection_module.GEOS5FPDayNotAvailable):
        connection.download_file(
            URL="https://example.test/2020/01/granule.nc4",
            filename=str(tmp_path / "granule.nc4"),
        )


def test_download_maps_missing_granule(tmp_path, monkeypatch):
    session = MagicMock()
    session.__enter__.return_value = session
    session.head.side_effect = [
        SimpleNamespace(status_code=404),
        SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
        ),
    ]
    monkeypatch.setattr(
        connection_module,
        "create_robust_session",
        lambda: session,
    )
    connection = connection_module.GEOS5FPConnection(
        download_directory=str(tmp_path),
    )

    with pytest.raises(connection_module.GEOS5FPGranuleNotAvailable):
        connection.download_file(
            URL="https://example.test/2020/01/granule.nc4",
            filename=str(tmp_path / "granule.nc4"),
        )
