"""
Create a legacy SSL context for older servers.

This module provides a function to create an SSL context that's more permissive
for connecting to older or misconfigured servers that don't support modern SSL/TLS.
"""

import ssl
from urllib3.util.ssl_ import create_urllib3_context


def create_legacy_ssl_context():
    """
    Create a legacy SSL context that's more permissive for older servers.
    
    This context uses reduced security settings including:
    - Lower security level (SECLEVEL=1)
    - Disabled hostname checking
    - Disabled certificate verification
    - Allowed legacy renegotiation
    
    Warning: This should only be used as a fallback when standard SSL fails.
    
    Returns
    -------
    ssl.SSLContext
        Configured legacy SSL context with reduced security settings.
    """
    context = create_urllib3_context()
    context.set_ciphers('DEFAULT@SECLEVEL=1')
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    # Allow legacy renegotiation
    context.options &= ~ssl.OP_NO_RENEGOTIATION
    return context
