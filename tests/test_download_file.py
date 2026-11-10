import importlib
from pathlib import Path
from unittest.mock import Mock


download_module = importlib.import_module("GEOS5FP.download_file")


def test_download_stages_chunks_outside_destination(tmp_path, monkeypatch):
    destination_dir = tmp_path / "destination"
    destination_dir.mkdir()
    destination = destination_dir / "granule.nc4"
    write_paths = []
    real_open = open

    def record_open(path, mode="r", *args, **kwargs):
        if mode == "wb":
            path = Path(path)
            write_paths.append(path)
            if destination_dir in path.parents:
                raise TimeoutError("simulated cloud-backed write timeout")
        return real_open(path, mode, *args, **kwargs)

    response = Mock()
    response.headers = {"content-length": "6"}
    response.iter_content.return_value = iter((b"GEOS", b"FP"))
    session = Mock()
    session.get.return_value = response
    monkeypatch.setattr(download_module, "open", record_open, raising=False)
    monkeypatch.setattr(download_module.requests, "Session", Mock(return_value=session))

    result = download_module.download_file(
        URL="https://example.test/granule.nc4",
        filename=str(destination),
        retries=1,
        wait_seconds=0,
    )

    assert result == str(destination)
    assert destination.read_bytes() == b"GEOSFP"
    assert len(write_paths) == 1
    assert destination_dir not in write_paths[0].parents
    assert not write_paths[0].exists()
