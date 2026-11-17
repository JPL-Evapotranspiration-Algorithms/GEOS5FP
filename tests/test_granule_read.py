import os
from unittest.mock import patch
import netCDF4
import numpy as np
import pytest
from rasterio.errors import RasterioIOError

from GEOS5FP.GEOS5FP_granule import GEOS5FPGranule
from GEOS5FP.exceptions import GEOS5FPGranuleNotAvailable


def _create_synthetic_geos5fp_nc4(path: str, var_name: str = "TOTEXTTAU") -> None:
    with netCDF4.Dataset(path, mode="w", format="NETCDF4") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("lat", 4)
        ds.createDimension("lon", 6)

        lat_var = ds.createVariable("lat", "f8", ("lat",))
        lon_var = ds.createVariable("lon", "f8", ("lon",))
        lat_var[:] = np.array([-45.0, -15.0, 15.0, 45.0])  # South-to-North
        lon_var[:] = np.array([-150.0, -90.0, -30.0, 30.0, 90.0, 150.0])

        var = ds.createVariable(var_name, "f4", ("time", "lat", "lon"), fill_value=1e15)
        data = np.arange(24, dtype=np.float32).reshape(1, 4, 6) * 0.01
        var[:] = data


def test_granule_read_falls_back_to_netcdf4_when_gdal_hdf5_missing(tmp_path):
    nc_file = tmp_path / "GEOS.fp.asm.tavg3_2d_aer_Nx.20260826_0130.V01.nc4"
    _create_synthetic_geos5fp_nc4(str(nc_file), var_name="TOTEXTTAU")

    granule = GEOS5FPGranule(str(nc_file))

    with patch(
        "rasters.Raster.open",
        side_effect=RasterioIOError(
            "Unable to open file as NetCDF: not recognized as being in a supported file format. "
            "It could have been recognized by driver HDF5, but plugin gdal_HDF5.so is not available in your installation."
        ),
    ):
        raster = granule.read("TOTEXTTAU")

    # File must NOT be deleted when fallback succeeds
    assert nc_file.exists()
    assert raster.shape == (4, 6)
    # South-to-North row 0 (0.00..0.05) should now be at the bottom row (-1)
    np.testing.assert_allclose(np.asarray(raster)[-1, :], np.arange(6) * 0.01, rtol=1e-5)


def test_granule_read_removes_genuinely_corrupted_file(tmp_path):
    corrupt_file = tmp_path / "GEOS.fp.asm.tavg3_2d_aer_Nx.20260826_0130.V01.nc4"
    corrupt_file.write_bytes(b"corrupted binary content")

    granule = GEOS5FPGranule(str(corrupt_file))

    with pytest.raises(GEOS5FPGranuleNotAvailable):
        granule.read("TOTEXTTAU")

    assert not corrupt_file.exists()
