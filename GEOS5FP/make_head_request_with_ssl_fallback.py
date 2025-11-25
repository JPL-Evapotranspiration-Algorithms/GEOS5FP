"""
Make HEAD request with SSL error handling and fallback strategies.

This module provides a function to make HTTP HEAD requests with multiple
fallback strategies for handling SSL/TLS connection issues.
"""

import logging
import warnings
import ssl

import requests
from requests.exceptions import SSLError
from requests.adapters import HTTPAdapter

from .create_robust_session import create_robust_session
from .create_legacy_ssl_context import create_legacy_ssl_context


def make_head_request_with_ssl_fallback(url, timeout=30):
    """
    Make HEAD request with SSL error handling and multiple fallback strategies.
    
    This function attempts to make a HEAD request using three strategies in order:
    1. Default SSL settings with retry strategy
    2. Legacy SSL context with reduced security
    3. Minimal SSL with shorter timeout
    
    Parameters
    ----------
    url : str
        The URL to make the HEAD request to.
    timeout : int, optional
        Timeout in seconds for the request. Default is 30.
    
    Returns
    -------
    requests.Response
        The response object from the successful HEAD request.
    
    Raises
    ------
    requests.exceptions.SSLError
        If all SSL fallback strategies fail to connect.
    """
    logger = logging.getLogger(__name__)
    
    # Strategy 1: Default SSL settings
    try:
        logger.debug(f"attempting HEAD request with default SSL: {url}")
        session = create_robust_session()
        return session.head(url, timeout=timeout, verify=True)
    except SSLError as e:
        logger.warning(f"SSL error with default settings: {e}")
        
        # Strategy 2: Legacy SSL context with reduced security
        try:
            logger.warning(f"retrying HEAD request with legacy SSL context: {url}")
            legacy_context = create_legacy_ssl_context()
            session = create_robust_session(ssl_context=legacy_context)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return session.head(url, timeout=timeout, verify=False)
        except SSLError as fallback_e:
            logger.warning(f"Legacy SSL context failed: {fallback_e}")
            
            # Strategy 3: Minimal SSL with shorter timeout
            try:
                logger.warning(f"retrying HEAD request with minimal SSL and shorter timeout: {url}")
                session = requests.Session()
                # Disable retries for this attempt to fail faster
                adapter = HTTPAdapter(max_retries=0)
                session.mount("https://", adapter)
                session.mount("http://", adapter)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return session.head(url, timeout=10, verify=False)
            except Exception as final_e:
                logger.error(f"All SSL fallback strategies failed. Final error: {final_e}")
                raise SSLError(f"Failed to connect to {url} after trying multiple SSL strategies. Original error: {e}")
        except Exception as fallback_e:
            logger.error(f"HEAD request failed with legacy SSL context: {fallback_e}")
            raise SSLError(f"Failed to connect to {url} due to SSL issues. Original error: {e}")
    except Exception as e:
        logger.error(f"HEAD request failed: {e}")
        raise
