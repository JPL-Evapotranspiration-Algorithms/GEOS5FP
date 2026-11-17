"""Tests for variable information lookup and queries."""
import pytest
from datetime import datetime
from unittest.mock import patch
import pandas as pd
import numpy as np
from shapely.geometry import Point, MultiPoint

from GEOS5FP import GEOS5FPConnection
from GEOS5FP.constants import GEOS5FP_VARIABLES, COMPUTED_VARIABLES
from GEOS5FP.GEOS5FP_point import PointQueryResult


@pytest.fixture
def conn():
    """Create a connection for testing."""
    return GEOS5FPConnection()


class TestGetVariableInfo:
    def test_get_variable_info_exists(self, conn):
        assert hasattr(conn, '_get_variable_info')
        assert callable(conn._get_variable_info)

    def test_get_variable_info_sfmc(self, conn):
        description, product, variable = conn._get_variable_info("SFMC")
        assert product == "tavg1_2d_lnd_Nx"
        assert variable == "SFMC"

    def test_get_variable_info_lai(self, conn):
        description, product, variable = conn._get_variable_info("LAI")
        assert product == "tavg1_2d_lnd_Nx"
        assert variable == "LAI"

    def test_get_variable_info_ta_k(self, conn):
        description, product, variable = conn._get_variable_info("Ta_K")
        assert product == "tavg1_2d_slv_Nx"
        assert variable == "T2M"

    def test_get_variable_info_invalid_raises(self, conn):
        with pytest.raises(KeyError):
            conn._get_variable_info("INVALID_VARIABLE_NAME")

    def test_get_variable_info_case_sensitive(self, conn):
        conn._get_variable_info("SFMC")
        with pytest.raises(KeyError):
            conn._get_variable_info("sfmc")


class TestVariableAliases:
    def test_sm_sfmc_alias(self, conn):
        assert conn._get_variable_info("SM") == conn._get_variable_info("SFMC")

    def test_ta_ta_k_alias(self, conn):
        assert conn._get_variable_info("Ta") == conn._get_variable_info("Ta_K")

    def test_ts_ts_k_alias(self, conn):
        assert conn._get_variable_info("Ts") == conn._get_variable_info("Ts_K")

    def test_vapor_alias(self, conn):
        assert conn._get_variable_info("vapor_kgsqm") == conn._get_variable_info("vapor_gccm")

    def test_ozone_alias(self, conn):
        assert conn._get_variable_info("ozone_dobson") == conn._get_variable_info("ozone_cm")


class TestAllVariablesAccessible:
    def test_all_variables_have_info(self, conn):
        for var_name in GEOS5FP_VARIABLES.keys():
            info = conn._get_variable_info(var_name)
            assert info is not None
            assert len(info) == 3

    def test_computed_albedo_variables_registered(self):
        assert "albedo_visible" in COMPUTED_VARIABLES
        assert "albedo_NIR" in COMPUTED_VARIABLES


class TestQueryBackwardCompatibility:
    """Tests for snapshot query routing compatibility and MultiPoint handling."""

    def test_query_routes_single_raster_snapshot_to_variable(self, conn, monkeypatch):
        sentinel = object()
        captured = {}

        def fake_variable(**kwargs):
            captured.update(kwargs)
            return sentinel

        monkeypatch.setattr(conn, 'variable', fake_variable)

        result = conn.query(
            target_variables="AOT",
            time_UTC=datetime(2024, 1, 1, 12, 0),
            geometry=object(),
            resampling="nearest",
            expected_hours=[1.5, 4.5]
        )

        assert result is sentinel
        assert captured["variable_name"] == "AOT"
        assert captured["resampling"] == "nearest"
        assert captured["expected_hours"] == [1.5, 4.5]

    def test_query_routes_raw_single_raster_snapshot_to_interpolate(self, conn, monkeypatch):
        sentinel = object()
        captured = {}

        def fake_interpolate(**kwargs):
            captured.update(kwargs)
            return sentinel

        monkeypatch.setattr(conn, 'interpolate', fake_interpolate)

        result = conn.query(
            target_variables="TESTVAR",
            dataset="inst3_2d_asm_Nx",
            time_UTC=datetime(2024, 1, 1, 12, 0),
            geometry=object()
        )

        assert result is sentinel
        assert captured["product"] == "inst3_2d_asm_Nx"
        assert captured["variable"] == "TESTVAR"

    def test_multipoint_single_time_queries_all_points(self, conn):
        points = MultiPoint([(-118.25, 34.05), (-122.42, 37.77), (-73.99, 40.75)])
        target_time = datetime(2024, 11, 15, 12, 0)

        def fake_point_query(dataset, variable, lat, lon, time_range, dropna=True):
            df = pd.DataFrame({variable: [round(abs(lon), 2)]}, index=[target_time])
            return PointQueryResult(data=None, df=df, lat_used=lat, lon_used=lon, url="mock://opendap")

        with patch("GEOS5FP.query.query_geos5fp_point", side_effect=fake_point_query):
            values = conn.AOT(time_UTC=target_time, geometry=points)

        assert isinstance(values, np.ndarray)
        assert len(values) == 3
        np.testing.assert_allclose(values, [118.25, 122.42, 73.99])
