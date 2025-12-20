#!/usr/bin/env python
"""
Test the fixed GEOS5FP point query with actual usage.
"""

import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

# Test the actual query_geos5fp_point function
logger.info("Testing GEOS5FP point query...")

from GEOS5FP.GEOS5FP_point import query_geos5fp_point

# Test parameters
dataset = "tavg1_2d_rad_Nx"
variable = "albedo"
lat = 34.0
lon = -118.0
time_start = datetime(2020, 6, 1, 0, 0)
time_end = datetime(2020, 6, 1, 3, 0)

logger.info(f"Querying {variable} from {dataset}")
logger.info(f"Location: ({lat}, {lon})")
logger.info(f"Time range: {time_start} to {time_end}")

try:
    result = query_geos5fp_point(
        dataset=dataset,
        variable=variable,
        lat=lat,
        lon=lon,
        time_range=(time_start, time_end),
        retries=3,
        retry_delay=10.0
    )
    
    logger.info("✓ Query successful!")
    logger.info(f"  URL: {result.url}")
    logger.info(f"  Grid lat used: {result.lat_used}")
    logger.info(f"  Grid lon used: {result.lon_used}")
    logger.info(f"  Data shape: {result.data.shape}")
    logger.info(f"  DataFrame shape: {result.df.shape}")
    logger.info(f"  DataFrame:\n{result.df}")
    
except Exception as e:
    logger.error(f"✗ Query failed: {e}", exc_info=True)
    sys.exit(1)
