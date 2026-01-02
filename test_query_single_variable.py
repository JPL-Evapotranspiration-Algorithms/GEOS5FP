import pytest
import pandas as pd
from shapely.geometry import Point
from GEOS5FP.GEOS5FP_connection import GEOS5FPConnection

@pytest.fixture
def geos5fp_connection():
    return GEOS5FPConnection()

def test_query_single_variable_non_raster_geometry(geos5fp_connection):
    # Mock inputs
    target_variable = "Ta_K"  # Use a predefined variable instead of raw name
    time_UTC = "2025-12-29 12:00:00"
    geometry = Point(-118.25, 34.05)  # Non-RasterGeometry

    # Call the query method
    result = geos5fp_connection.query(
        target_variables=target_variable,
        time_UTC=time_UTC,
        geometry=geometry
    )

    # Assert the result is a DataFrame (point queries return DataFrame per docstring)
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame for point queries"
    assert 'Ta_K' in result.columns, "Result should contain the queried variable"

    # Additional checks (e.g., shape, values) can be added based on expected output