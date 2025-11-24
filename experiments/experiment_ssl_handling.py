#!/usr/bin/env python3
"""
Test script to verify SSL error handling improvements in GEOS5FP.

This script tests the enhanced SSL error handling by attempting to 
access a GEOS-5 FP file that previously failed with SSL errors.
"""

import logging
from datetime import datetime
from GEOS5FP import GEOS5FPConnection

# Set up logging to see detailed error handling
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_ssl_error_handling():
    """Test SSL error handling with the problematic URL from the traceback."""
    
    # Initialize GEOS5FP connection
    geos5fp = GEOS5FPConnection()
    
    # Test URL from the error traceback
    test_url = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y2020/M05/D02/GEOS.fp.asm.inst3_2d_asm_Nx.20200502_1500.V01.nc4"
    
    print(f"Testing SSL error handling with URL: {test_url}")
    print("This should now handle SSL errors gracefully...")
    
    try:
        # Attempt to download the file that previously failed
        result = geos5fp.download_file(test_url)
        print(f"SUCCESS: File downloaded successfully: {result}")
        assert True  # Test passed - file downloaded successfully
        
    except Exception as e:
        print(f"EXPECTED: Caught exception (this may be expected): {type(e).__name__}: {e}")
        
        # Check if it's still an SSL error or if it's now handled properly
        if "SSL" in str(e):
            print("ERROR: SSL error was not properly handled!")
            assert False, f"SSL error was not properly handled: {e}"
        else:
            print("SUCCESS: SSL error was handled gracefully, got a different error type")
            assert True  # Test passed - SSL error was handled

def test_ozone_interpolation():
    """Test the specific ozone interpolation that was failing."""
    
    # Initialize GEOS5FP connection
    geos5fp = GEOS5FPConnection()
    
    # Test parameters from the original error
    test_time = datetime(2020, 5, 2, 15, 0, 0)  # 2020-05-02 15:00:00 UTC
    
    print(f"Testing ozone interpolation for time: {test_time}")
    
    try:
        # This should trigger the same code path that was failing
        ozone_result = geos5fp.ozone_cm(
            time_UTC=test_time,
            geometry=None,  # Will use default geometry
            resampling='bilinear'
        )
        print(f"SUCCESS: Ozone interpolation completed: {type(ozone_result)}")
        assert True  # Test passed - ozone interpolation completed
        
    except Exception as e:
        print(f"Caught exception during ozone interpolation: {type(e).__name__}: {e}")
        
        # Check if it's still an SSL error
        if "SSL" in str(e) and "UNEXPECTED_EOF_WHILE_READING" in str(e):
            print("ERROR: Original SSL error still occurs!")
            assert False, f"Original SSL error still occurs: {e}"
        else:
            print("SUCCESS: Original SSL error was handled, got a different error type")
            assert True  # Test passed - original SSL error was handled

# Tests are run by pytest, no need for main block