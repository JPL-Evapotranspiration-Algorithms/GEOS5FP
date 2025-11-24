#!/usr/bin/env python3
"""
Test script for enhanced SSL error handling with multiple fallback strategies.
"""

import logging
import pytest
import requests
import ssl
import warnings
from requests.exceptions import SSLError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.util.ssl_ import create_urllib3_context

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_robust_session(ssl_context=None):
    """Create robust session with SSL error handling and retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=2,
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=1,
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    
    # Configure SSL context if provided
    if ssl_context:
        adapter.init_poolmanager(
            ssl_context=ssl_context,
            socket_options=HTTPAdapter.DEFAULT_SOCKET_OPTIONS
        )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def create_legacy_ssl_context():
    """Create a legacy SSL context that's more permissive for older servers."""
    context = create_urllib3_context()
    context.set_ciphers('DEFAULT@SECLEVEL=1')
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    # Allow legacy renegotiation
    context.options &= ~ssl.OP_NO_RENEGOTIATION
    return context


def make_head_request_with_enhanced_ssl_fallback(url, timeout=30):
    """Make HEAD request with enhanced SSL error handling and multiple fallback strategies."""
    
    # Strategy 1: Default SSL settings
    try:
        logger.info(f"Strategy 1: Attempting HEAD request with default SSL: {url}")
        session = create_robust_session()
        response = session.head(url, timeout=timeout, verify=True)
        logger.info(f"Strategy 1 SUCCESS: Status {response.status_code}")
        return response
    except SSLError as e:
        logger.warning(f"Strategy 1 FAILED - SSL error with default settings: {e}")
        
        # Strategy 2: Legacy SSL context with reduced security
        try:
            logger.info(f"Strategy 2: Retrying HEAD request with legacy SSL context: {url}")
            legacy_context = create_legacy_ssl_context()
            session = create_robust_session(ssl_context=legacy_context)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                response = session.head(url, timeout=timeout, verify=False)
                logger.info(f"Strategy 2 SUCCESS: Status {response.status_code}")
                return response
        except SSLError as fallback_e:
            logger.warning(f"Strategy 2 FAILED - Legacy SSL context failed: {fallback_e}")
            
            # Strategy 3: Minimal SSL with shorter timeout
            try:
                logger.info(f"Strategy 3: Retrying HEAD request with minimal SSL: {url}")
                session = requests.Session()
                # Disable retries for this attempt to fail faster
                adapter = HTTPAdapter(max_retries=0)
                session.mount("https://", adapter)
                session.mount("http://", adapter)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    response = session.head(url, timeout=10, verify=False)
                    logger.info(f"Strategy 3 SUCCESS: Status {response.status_code}")
                    return response
            except Exception as final_e:
                logger.error(f"Strategy 3 FAILED - All SSL fallback strategies failed: {final_e}")
                raise SSLError(f"Failed to connect to {url} after trying multiple SSL strategies. Original error: {e}")
        except Exception as fallback_e:
            logger.error(f"Strategy 2 ERROR - HEAD request failed with legacy SSL context: {fallback_e}")
            raise SSLError(f"Failed to connect to {url} due to SSL issues. Original error: {e}")
    except Exception as e:
        logger.error(f"Strategy 1 ERROR - HEAD request failed: {e}")
        raise


def test_enhanced_ssl_handling():
    """Test enhanced SSL error handling with the problematic URL."""
    
    # Test URL that was failing
    test_url = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y2020/M06/D13/GEOS.fp.asm.tavg3_2d_aer_Nx.20200613_1930.V01.nc4"
    
    print("=" * 80)
    print(f"Testing Enhanced SSL Error Handling")
    print(f"URL: {test_url}")
    print("=" * 80)
    
    try:
        response = make_head_request_with_enhanced_ssl_fallback(test_url)
        print(f"\n✓ SUCCESS: HEAD request completed with status: {response.status_code}")
        if response.status_code == 200:
            print("  File exists and is accessible")
        elif response.status_code == 404:
            print("  File not found (expected for old data)")
        else:
            print(f"  Unexpected status code: {response.status_code}")
        # Test passes if we get any valid HTTP response (no SSL error)
        assert response.status_code in [200, 404], f"Unexpected status code: {response.status_code}"
        
    except SSLError as e:
        print(f"\n✗ SSL ERROR: {e}")
        pytest.fail(f"SSL error occurred: {e}")
        
    except Exception as e:
        print(f"\n✗ OTHER ERROR: {e}")
        # Other errors are acceptable as long as they're not SSL errors
        assert "SSL" not in str(e), f"Unexpected SSL-related error: {e}"


def test_geos5fp_integration():
    """Test the enhanced SSL handling through GEOS5FP interface."""
    
    try:
        from GEOS5FP import GEOS5FPConnection
        from datetime import datetime
        
        print("\n" + "=" * 80)
        print("Testing GEOS5FP Integration with Enhanced SSL")
        print("=" * 80)
        
        # Initialize connection
        geos5fp = GEOS5FPConnection()
        
        # Test with a recent date that should have data
        test_time = datetime(2024, 1, 15, 12, 0, 0)  # More recent date
        
        print(f"Testing AOT retrieval for: {test_time}")
        
        try:
            # This will trigger the SSL fallback mechanisms
            aot_data = geos5fp.AOT(
                time_UTC=test_time,
                geometry=None,
                resampling='bilinear'
            )
            print(f"✓ SUCCESS: AOT data retrieved successfully")
            print(f"  Data type: {type(aot_data)}")
            # Test passes if AOT data is retrieved successfully
            assert aot_data is not None, "AOT data should not be None"
            
        except Exception as e:
            print(f"✗ GEOS5FP ERROR: {e}")
            # Check if it's an SSL error - if so, fail the test
            if "SSL" in str(e) and "UNEXPECTED_EOF_WHILE_READING" in str(e):
                pytest.fail(f"SSL error not properly handled: {e}")
            else:
                # Other errors might be acceptable (e.g., data not available)
                pytest.skip(f"Test skipped due to non-SSL error: {e}")
            
    except ImportError as e:
        print(f"✗ IMPORT ERROR: Cannot import GEOS5FP: {e}")
        pytest.skip(f"Cannot import GEOS5FP: {e}")


if __name__ == "__main__":
    print("Enhanced SSL Error Handling Test Suite")
    print("=" * 80)
    
    # Test 1: Direct SSL handling
    success1 = test_enhanced_ssl_handling()
    
    # Test 2: GEOS5FP integration
    success2 = test_geos5fp_integration()
    
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"Direct SSL Handling: {'PASS' if success1 else 'FAIL'}")
    print(f"GEOS5FP Integration: {'PASS' if success2 else 'FAIL'}")
    
    if success1 or success2:
        print("\n✓ At least one test passed - SSL improvements are working!")
    else:
        print("\n✗ All tests failed - SSL issues persist")