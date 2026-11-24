"""Tests for variable information lookup and queries."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, MultiPoint

from GEOS5FP import GEOS5FPConnection, query
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

    def test_computed_albedo_and_td_c_variables_registered(self):
        assert "albedo_visible" in COMPUTED_VARIABLES
        assert "albedo_NIR" in COMPUTED_VARIABLES
        assert "Td_C" in COMPUTED_VARIABLES


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

    def test_query_single_element_list_and_return_dict_on_raster(self, conn, monkeypatch):
        sentinel = object()
        captured = {}

        def fake_variable(**kwargs):
            captured.update(kwargs)
            return sentinel

        monkeypatch.setattr(conn, 'variable', fake_variable)

        result = conn.query(
            variables=["CO2SC"],
            time_UTC=datetime(2024, 10, 22, 18, 30),
            geometry=object()
        )
        assert result is sentinel
        assert captured["variable_name"] == "CO2SC"

        dict_result = conn.query(
            variables=["CO2SC"],
            time_UTC=datetime(2024, 10, 22, 18, 30),
            geometry=object(),
            return_dict=True,
        )
        assert dict_result == {"CO2SC": sentinel}

    def test_query_empty_variable_list_raises(self, conn):
        with pytest.raises(ValueError, match="At least one variable must be specified"):
            conn.query(
                variables=[],
                time_UTC=datetime(2024, 10, 22, 18, 30),
                geometry=object()
            )

    def test_query_multiple_variables_on_raster_returns_dict(self, conn, monkeypatch):
        def fake_variable(**kwargs):
            return f"raster:{kwargs['variable_name']}"

        monkeypatch.setattr(conn, 'variable', fake_variable)

        result = conn.query(
            variables=["CO2SC", "Ta_K"],
            time_UTC=datetime(2024, 10, 22, 18, 30),
            geometry=object()
        )

        assert result == {"CO2SC": "raster:CO2SC", "Ta_K": "raster:Ta_K"}

    def test_standalone_query_raster_snapshot(self, monkeypatch):
        sentinel = object()

        def fake_variable(self, **kwargs):
            assert kwargs["variable_name"] == "CO2SC"
            return sentinel

        monkeypatch.setattr(GEOS5FPConnection, 'variable', fake_variable)

        result = query(
            ["CO2SC"],
            time_UTC=datetime(2024, 10, 22, 18, 30),
            geometry=object()
        )
        assert result is sentinel

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

    def test_variable_method_routes_computed_variables_and_deprecated_wind_speed(self, conn, monkeypatch):
        monkeypatch.setattr(conn, "wind_speed_mps", lambda **kwargs: 4.5)
        monkeypatch.setattr(conn, "Td_K", lambda **kwargs: 285.15)

        assert conn.variable("wind_speed_mps", time_UTC=datetime(2024, 11, 15, 12, 0), geometry=object()) == 4.5
        assert conn.Td_C(time_UTC=datetime(2024, 11, 15, 12, 0), geometry=object()) == pytest.approx(12.0)

        with pytest.warns(DeprecationWarning, match="wind_speed\(\) is deprecated"):
            assert conn.wind_speed(time_UTC=datetime(2024, 11, 15, 12, 0), geometry=object()) == 4.5

    def test_vapor_gccm_and_ozone_cm_raster_unit_conversions(self, conn, monkeypatch):
        """Regression test for issue #83: gridded raster queries must apply unit scaling."""
        raw_values = {
            "TQV": np.array([[10.718, 25.0]], dtype=np.float64),
            "TO3": np.array([[334.4433, 300.0]], dtype=np.float64),
        }

        def fake_interpolate(time_UTC, product, variable, geometry=None, **kwargs):
            assert product == "inst3_2d_asm_Nx"
            return raw_values[variable].copy()

        monkeypatch.setattr(conn, "interpolate", fake_interpolate)

        grid = object()
        t = datetime(2026, 6, 2, 18, 0)

        vapor_raster = conn.vapor_gccm(time_UTC=t, geometry=grid)
        ozone_raster = conn.ozone_cm(time_UTC=t, geometry=grid)
        vapor_kgsqm_raster = conn.vapor_kgsqm(time_UTC=t, geometry=grid)
        ozone_dobson_raster = conn.ozone_dobson(time_UTC=t, geometry=grid)

        np.testing.assert_allclose(vapor_raster, [[1.0718, 2.5]])
        np.testing.assert_allclose(ozone_raster, [[0.3344433, 0.3]])
        np.testing.assert_allclose(vapor_kgsqm_raster, [[10.718, 25.0]])
        np.testing.assert_allclose(ozone_dobson_raster, [[334.4433, 300.0]])

        multi_rasters = conn.query(
            target_variables=["vapor_gccm", "ozone_cm"],
            time_UTC=t,
            geometry=grid,
        )
        np.testing.assert_allclose(multi_rasters["vapor_gccm"], [[1.0718, 2.5]])
        np.testing.assert_allclose(multi_rasters["ozone_cm"], [[0.3344433, 0.3]])

    def test_vapor_gccm_and_ozone_cm_point_query_no_double_scaling(self, conn):
        """Ensure OPeNDAP point queries still apply unit scaling exactly once."""
        target_time = datetime(2026, 6, 2, 18, 0)
        point = Point(-118.25, 34.05)

        def fake_point_query(dataset, variable, lat, lon, time_range, dropna=True):
            raw = 10.718 if variable == "tqv" else 334.4433
            df = pd.DataFrame({variable: [raw]}, index=[target_time])
            return PointQueryResult(data=None, df=df, lat_used=lat, lon_used=lon, url="mock://opendap")

        with patch("GEOS5FP.query.query_geos5fp_point", side_effect=fake_point_query):
            vapor_pt = conn.vapor_gccm(time_UTC=target_time, geometry=point)
            ozone_pt = conn.ozone_cm(time_UTC=target_time, geometry=point)

        np.testing.assert_allclose(vapor_pt, [1.0718])
        np.testing.assert_allclose(ozone_pt, [0.3344433])

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

    def test_batch_query_multipoint_unpacks_distinct_coordinates_and_validates_length(self, conn):
        times = [
            datetime(2024, 11, 15, 12, 0),
            datetime(2024, 11, 15, 13, 0),
            datetime(2024, 11, 15, 14, 0),
        ]
        points = MultiPoint([(-118.25, 34.05), (-122.42, 37.77), (-73.99, 40.75)])
        queried_coords = []

        def fake_point_query(dataset, variable, lat, lon, time_range, dropna=True):
            queried_coords.append((lat, lon))
            mid = time_range[0] + (time_range[1] - time_range[0]) / 2
            df = pd.DataFrame({variable: [round(abs(lon), 2)]}, index=[mid])
            return PointQueryResult(data=None, df=df, lat_used=lat, lon_used=lon, url="mock://opendap")

        with patch("GEOS5FP.query.query_geos5fp_point", side_effect=fake_point_query):
            vals = conn.query(target_variables="AOT", time_UTC=times, geometry=points)

        assert queried_coords == [(34.05, -118.25), (37.77, -122.42), (40.75, -73.99)]
        np.testing.assert_allclose(vals, [118.25, 122.42, 73.99])

        with pytest.raises(ValueError, match="Number of points in MultiPoint"):
            conn.query(
                target_variables="AOT",
                time_UTC=times,
                geometry=MultiPoint([(-118.25, 34.05), (-122.42, 37.77)]),
            )

    def test_multipoint_time_range_preserves_point_index(self, conn):
        points = MultiPoint([(-118.25, 34.05), (-122.42, 37.77)])
        t0 = datetime(2024, 11, 15, 12, 0)
        t1 = datetime(2024, 11, 15, 13, 0)

        def fake_point_query(dataset, variable, lat, lon, time_range, dropna=True):
            val = 10.0 if lon == -118.25 else 20.0
            df = pd.DataFrame({variable: [val, val + 1.0]}, index=pd.DatetimeIndex([t0, t1], name="time"))
            return PointQueryResult(data=None, df=df, lat_used=lat, lon_used=lon, url="mock://opendap")

        with patch("GEOS5FP.query.query_geos5fp_point", side_effect=fake_point_query):
            res = conn.query(
                target_variables="AOT",
                time_range=(t0, t1),
                geometry=points,
            )

        assert isinstance(res, gpd.GeoDataFrame)
        assert "point_index" in res.columns
        assert list(res["point_index"]) == [0, 0, 1, 1]
        np.testing.assert_allclose(res["AOT"].values, [10.0, 11.0, 20.0, 21.0])

    def test_batch_query_all_computed_variables_and_interpolation(self, conn):
        t0 = datetime(2024, 11, 15, 12, 0)
        t1 = datetime(2024, 11, 15, 13, 0)
        target_time = datetime(2024, 11, 15, 12, 30)

        targets = gpd.GeoDataFrame(
            {"time_UTC": [target_time], "geometry": [Point(-118.25, 34.05)]},
            crs="EPSG:4326"
        )

        def fake_multi(dataset, variables, lat, lon, time_range, dropna=True):
            data = {
                "qv2m": [0.008, 0.010],
                "ps": [101000.0, 102000.0],
                "t2m": [290.0, 300.0],
                "t2mmin": [280.0, 282.0],
                "u2m": [3.0, 3.0],
                "v2m": [4.0, 4.0],
                "tqv": [20.0, 30.0],
                "to3": [300.0, 320.0],
                "albedo": [0.20, 0.20],
                "albvisdr": [0.05, 0.05],
                "albvisdf": [0.04, 0.04],
                "albnirdr": [0.10, 0.10],
                "albnirdf": [0.08, 0.08],
            }
            cols = {v: data[v] for v in variables}
            df = pd.DataFrame(cols, index=pd.DatetimeIndex([t0, t1], name="time"))
            return PointQueryResult(data=None, df=df, lat_used=lat, lon_used=lon, url="mock://opendap")

        def fake_single(dataset, variable, lat, lon, time_range, dropna=True):
            return fake_multi(dataset, [variable], lat, lon, time_range, dropna=dropna)

        all_vars = [
            "RH", "Ta_C", "Tmin_C", "SVP_Pa", "Ea_Pa", "VPD_kPa", "Td_K", "Td_C",
            "wind_speed_mps", "PAR_proportion", "NIR_proportion",
            "PAR_albedo", "NIR_albedo", "albedo_visible", "albedo_NIR",
            "vapor_gccm", "ozone_cm"
        ]

        with patch("GEOS5FP.query.query_geos5fp_point_multi", side_effect=fake_multi),              patch("GEOS5FP.query.query_geos5fp_point", side_effect=fake_single):
            out = conn.query(target_variables=all_vars, targets_df=targets)

        # Midpoint interpolation checks
        assert out["Ta_C"].iloc[0] == pytest.approx(295.0 - 273.15)
        assert out["Tmin_C"].iloc[0] == pytest.approx(281.0 - 273.15)
        assert out["wind_speed_mps"].iloc[0] == pytest.approx(5.0)
        assert out["vapor_gccm"].iloc[0] == pytest.approx(2.5)
        assert out["ozone_cm"].iloc[0] == pytest.approx(0.31)
        assert out["PAR_proportion"].iloc[0] == pytest.approx(0.25)
        assert out["NIR_proportion"].iloc[0] == pytest.approx(0.50)
        assert out["PAR_albedo"].iloc[0] == pytest.approx(0.09)
        assert out["NIR_albedo"].iloc[0] == pytest.approx(0.18)
        assert out["albedo_visible"].iloc[0] == pytest.approx(0.05)
        assert out["albedo_NIR"].iloc[0] == pytest.approx(0.10)
        assert 0.0 < out["RH"].iloc[0] < 1.0
        assert out["SVP_Pa"].iloc[0] > 0.0
        assert out["Ea_Pa"].iloc[0] > 0.0
        assert out["VPD_kPa"].iloc[0] > 0.0
        assert out["Td_K"].iloc[0] < 295.0
        assert out["Td_C"].iloc[0] == pytest.approx(out["Td_K"].iloc[0] - 273.15)
