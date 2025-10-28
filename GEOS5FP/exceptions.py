class GEOS5FPGranuleNotAvailable(Exception):
    pass


class GEOS5FPDayNotAvailable(Exception):
    pass


class GEOS5FPMonthNotAvailable(Exception):
    pass


class GEOS5FPYearNotAvailable(Exception):
    pass


class FailedGEOS5FPDownload(ConnectionError):
    pass


class GEOS5FPSSLError(Exception):
    """
    Exception raised when SSL/TLS connection errors occur during GEOS-5 FP data access.
    
    This typically happens due to:
    - SSL certificate verification issues
    - Network connectivity problems 
    - Server-side SSL configuration changes
    - Firewall or proxy interference
    
    The GEOS5FP library includes automatic SSL error handling and fallback mechanisms,
    but this exception is raised when all fallback attempts have been exhausted.
    """
    
    def __init__(self, message, original_error=None, url=None):
        self.original_error = original_error
        self.url = url
        
        detailed_message = f"SSL connection error"
        if url:
            detailed_message += f" for URL: {url}"
        detailed_message += f": {message}"
        
        if original_error:
            detailed_message += f" (Original error: {original_error})"
            
        detailed_message += "\n\nTroubleshooting suggestions:"
        detailed_message += "\n- Check your internet connection"
        detailed_message += "\n- Verify that portal.nccs.nasa.gov is accessible"
        detailed_message += "\n- Try again in a few minutes (server may be temporarily unavailable)"
        detailed_message += "\n- Check if your firewall or proxy is blocking HTTPS connections"
        
        super().__init__(detailed_message)
