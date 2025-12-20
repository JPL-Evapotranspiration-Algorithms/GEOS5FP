#!/usr/bin/env python
"""
Test script to diagnose OPeNDAP hanging issue.
"""

import logging
import sys
from datetime import datetime

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

# Test direct xarray access
logger.info("Testing direct xarray OPeNDAP access...")

import xarray as xr

url = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/tavg1_2d_rad_Nx"
logger.info(f"Opening: {url}")

try:
    import socket
    
    # Set socket timeout like in the fixed code
    logger.info("Setting socket timeout to 30 seconds")
    original_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(30.0)
    
    # Test with minimal options first
    ds = xr.open_dataset(
        url,
        engine='netcdf4',
        cache=False,
        lock=False,
        decode_cf=False
    )
    
    logger.info(f"Dataset opened successfully")
    logger.info(f"Available variables: {list(ds.data_vars)[:5]}...")
    logger.info(f"Dimensions: {dict(ds.dims)}")
    
    # Try a simple selection
    logger.info("Attempting point selection...")
    pt = ds.sel(lat=34.0, lon=-118.0, method='nearest')
    logger.info("Point selection successful")
    
    # Try accessing a single variable
    logger.info("Accessing 'albedo' variable...")
    if 'albedo' in pt:
        da = pt['albedo']
        logger.info(f"Variable shape: {da.shape}")
        logger.info(f"Variable dims: {da.dims}")
        
        # Try selecting a single time point
        logger.info("Selecting first time point...")
        da_single = da.isel(time=0)
        logger.info(f"Single time shape: {da_single.shape}")
        
        # Try getting the value
        logger.info("Getting value (this may hang)...")
        val = float(da_single.values)
        logger.info(f"✓ Got value: {val}")
    else:
        logger.warning("'albedo' not found in dataset")
    
    ds.close()
    socket.setdefaulttimeout(original_timeout)
    logger.info("✓ Test completed successfully")
    
except Exception as e:
    socket.setdefaulttimeout(original_timeout)
    logger.error(f"✗ Error: {e}", exc_info=True)
