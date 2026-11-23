"""Tests for OPeNDAP point query helpers and return types."""
from datetime import datetime
import socket
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, MultiPoint

from GEOS5FP import GEOS5FP, GEOS5FPConnection
from GEOS5FP.GEOS5FP_point import PointQueryResult, query_geos5fp_point_multi


def _mock_point_result(dataset, variable, lat, lon, time_range, dropna=True, **kwargs):
    start, end = time_range
    mid = start + (end - start) / 2
    df = pd.DataFrame({variable: [0.42]}, index=pd.DatetimeIndex([mid], name="time"))
    return PointQueryResult(data=None, df=df, lat_used=lat, lon_used=lon, url="mock://opendap")


def test_single_point_query_returns_numpy_array():
    conn = GEOS5FP()
    time_utc = datetime(2024, 11, 15, 12, 0, 0)
    geometry = Point(-118.25, 34.05)

    with patch("GEOS5FP.query.query_geos5fp_point", side_effect=_mock_point_result):
        result = conn.COT(time_UTC=time_utc, geometry=geometry)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)


def test_multipoint_query_returns_numpy_array():
    conn = GEOS5FP()
    time_utc = datetime(2024, 11, 15, 12, 0, 0)
    points = MultiPoint([
        Point(-118.25, 34.05),
        Point(-122.42, 37.77),
        Point(-73.99, 40.75),
    ])

    with patch("GEOS5FP.query.query_geos5fp_point", side_effect=_mock_point_result):
        result = conn.AOT(time_UTC=time_utc, geometry=points)

    assert isinstance(result, np.ndarray)
    assert len(result) == 3


def test_query_single_variable_non_raster_geometry_returns_array():
    conn = GEOS5FP()

    with patch("GEOS5FP.query.query_geos5fp_point", side_effect=_mock_point_result):
        result = conn.query(
            target_variables="Ta_K",
            time_UTC="2025-12-29 12:00:00",
            geometry=Point(-118.25, 34.05),
        )

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [0.42])


def test_single_point_query_allow_fallback_uses_interpolate():
    conn = GEOS5FPConnection(allow_fallback=True)
    target_time = datetime(2024, 11, 15, 12, 0)

    with patch("GEOS5FP.query.query_geos5fp_point", side_effect=TimeoutError("504 Gateway Time-out")), \
         patch.object(conn, "interpolate", return_value=np.array([295.5])) as mock_interp:
        result = conn.query(
            target_variables="Ta_C",
            time_UTC=target_time,
            geometry=Point(-118.25, 34.05),
        )

    mock_interp.assert_called_once()
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [295.5 - 273.15])


def test_query_geos5fp_point_multi_retries_and_restores_socket_timeout():
    original_timeout = socket.getdefaulttimeout()
    attempts = []

    def flaky_open_dataset(url, **kwargs):
        attempts.append(url)
        if len(attempts) == 1:
            raise TimeoutError("504 Gateway Time-out")
        raise RuntimeError("stop after retry verification")

    with patch("GEOS5FP.GEOS5FP_point.xr.open_dataset", side_effect=flaky_open_dataset):
        with pytest.raises(RuntimeError, match="stop after retry verification"):
            query_geos5fp_point_multi(
                dataset="tavg1_2d_slv_Nx",
                variables=["t2m", "ps"],
                lat=34.05,
                lon=-118.25,
                retries=2,
                retry_delay=0.0,
                timeout=99.0,
            )

    assert len(attempts) == 2
    assert socket.getdefaulttimeout() == original_timeout
