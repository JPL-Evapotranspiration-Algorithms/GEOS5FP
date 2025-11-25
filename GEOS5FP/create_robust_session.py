"""
Create a robust HTTP session with SSL error handling and retry strategy.

This module provides a function to create a requests Session configured with
automatic retries for transient failures and optional custom SSL context.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_robust_session(ssl_context=None):
    """
    Create robust session with SSL error handling and retry strategy.
    
    Parameters
    ----------
    ssl_context : ssl.SSLContext, optional
        Custom SSL context to use for the session. If None, default SSL settings are used.
    
    Returns
    -------
    requests.Session
        Configured session with retry strategy and optional SSL context.
    """
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
