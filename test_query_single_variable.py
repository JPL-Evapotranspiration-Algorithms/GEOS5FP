import pytest
import numpy as np
from shapely.geometry import Point
from GEOS5FP.GEOS5FP_connection import GEOS5FPConnection

@pytest.fixture
def geos5fp_connection():
    return GEOS5FPConnection()

def test_query_single_variable_non_raster_geometry(geos5fp_connection):
    # Mock inputs
    target_variable = "T2M"
    time_UTC = "2025-12-29 12:00:00"
    geometry = Point(-118.25, 34.05)  # Non-RasterGeometry

    # Call the query method
    result = geos5fp_connection.query(
        target_variables=target_variable,
        time_UTC=time_UTC,
        geometry=geometry
    )

    # Assert the result is a numpy array
    assert isinstance(result, np.ndarray), "Result should be a numpy array"

    # Additional checks (e.g., shape, values) can be added based on expected output