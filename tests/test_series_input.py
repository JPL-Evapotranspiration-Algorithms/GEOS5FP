"""
Tests for handling pandas Series and array-like time_UTC inputs.
"""

from datetime import datetime
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from GEOS5FP import GEOS5FP
from GEOS5FP.GEOS5FP_point import PointQueryResult


def _mock_point_query(dataset, variable, lat, lon, time_range, dropna=True):
    start, end = time_range
    mid = start + (end - start) / 2
    df = pd.DataFrame({variable: [0.25]}, index=[mid])
    return PointQueryResult(data=None, df=df, lat_used=lat, lon_used=lon, url="mock://opendap")


class TestSeriesTimeInput:
    """Test handling of pandas Series time inputs."""

    @pytest.fixture
    def connection(self):
        return GEOS5FP(download_directory="test_download")

    def test_series_with_geometry_array(self, connection):
        times = pd.Series([
            datetime(2019, 1, 28, 21, 8, 10),
            datetime(2019, 1, 28, 22, 40, 54)
        ])
        geometries = [
            Point(-82.2188, 29.7381),
            Point(-106.8425, 32.5907)
        ]

        with patch("GEOS5FP.query.query_geos5fp_point", side_effect=_mock_point_query):
            result = connection.COT(time_UTC=times, geometry=geometries)

        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_series_with_single_geometry(self, connection):
        times = pd.Series([
            datetime(2019, 1, 28, 21, 0, 0),
            datetime(2019, 1, 28, 22, 0, 0)
        ])
        geometry = Point(-82.2188, 29.7381)

        with patch("GEOS5FP.query.query_geos5fp_point", side_effect=_mock_point_query):
            result = connection.COT(time_UTC=times, geometry=geometry)

        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_list_time_input(self, connection):
        times = [
            datetime(2019, 1, 28, 21, 0, 0),
            datetime(2019, 1, 28, 22, 0, 0)
        ]
        geometry = Point(-82.2188, 29.7381)

        with patch("GEOS5FP.query.query_geos5fp_point", side_effect=_mock_point_query):
            result = connection.COT(time_UTC=times, geometry=geometry)

        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_tuple_time_input(self, connection):
        times = (
            datetime(2019, 1, 28, 21, 0, 0),
            datetime(2019, 1, 28, 22, 0, 0)
        )
        geometry = Point(-82.2188, 29.7381)

        with patch("GEOS5FP.query.query_geos5fp_point", side_effect=_mock_point_query):
            result = connection.COT(time_UTC=times, geometry=geometry)

        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_numpy_array_time_input(self, connection):
        times = np.array([
            datetime(2019, 1, 28, 21, 0, 0),
            datetime(2019, 1, 28, 22, 0, 0)
        ])
        geometry = Point(-82.2188, 29.7381)

        with patch("GEOS5FP.query.query_geos5fp_point", side_effect=_mock_point_query):
            result = connection.COT(time_UTC=times, geometry=geometry)

        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_mismatched_lengths(self, connection):
        times = pd.Series([
            datetime(2019, 1, 28, 21, 0, 0),
            datetime(2019, 1, 28, 22, 0, 0)
        ])
        geometries = [
            Point(-82.2188, 29.7381)
        ]

        with pytest.raises(
            ValueError,
            match=r"Number of times \(\d+\) must match number of geometries \(\d+\)",
        ):
            connection.COT(time_UTC=times, geometry=geometries)

    def test_single_time_still_works(self, connection):
        time = datetime(2019, 1, 28, 21, 0, 0)
        geometry = Point(-82.2188, 29.7381)

        with patch("GEOS5FP.query.query_geos5fp_point", side_effect=_mock_point_query):
            result = connection.COT(time_UTC=time, geometry=geometry)

        assert isinstance(result, np.ndarray)
        assert len(result) == 1
