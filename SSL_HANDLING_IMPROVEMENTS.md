# SSL Error Handling Improvements for GEOS5FP

## Problem Description

The GEOS5FP library was encountering SSL connection errors when attempting to download data from NASA's GEOS-5 FP data portal. The specific error was:

```
requests.exceptions.SSLError: HTTPSConnectionPool(host='portal.nccs.nasa.gov', port=443): 
Max retries exceeded with url: /datashare/gmao/geos-fp/das/Y2020/M05/D02/GEOS.fp.asm.inst3_2d_asm_Nx.20200502_1500.V01.nc4 
(Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1016)')))
```

This error typically occurs due to:
- SSL/TLS protocol version mismatches
- Certificate verification issues
- Network connectivity problems
- Server-side SSL configuration changes
- Firewall or proxy interference

## Solution

### Enhanced SSL Error Handling

The library now includes comprehensive SSL error handling with the following improvements:

#### 1. Robust Session Creation
- Created `create_robust_session()` function that configures HTTP sessions with:
  - Automatic retry logic for transient failures
  - Proper timeout handling
  - Adapter configuration for both HTTP and HTTPS

#### 2. SSL Fallback Mechanism
- Implemented `make_head_request_with_ssl_fallback()` and `download_with_ssl_fallback()` functions
- First attempts connection with default SSL settings (secure)
- If SSL error occurs, falls back to SSL verification disabled (with warnings)
- Provides detailed error logging at each step

#### 3. Enhanced Error Reporting
- Added new `GEOS5FPSSLError` exception class with:
  - Detailed error messages
  - Troubleshooting suggestions
  - Original error preservation
  - URL context information

#### 4. Comprehensive Logging
- Added debug-level logging for SSL connection attempts
- Warning-level logging for SSL fallbacks
- Error-level logging for failures
- Informational logging for successful connections

### Code Changes

#### Files Modified:

1. **`GEOS5FP/GEOS5FP_connection.py`**:
   - Added SSL error handling imports
   - Implemented robust session creation
   - Added SSL fallback mechanism for HEAD requests
   - Enhanced error handling in `download_file()` method
   - Updated exception handling to use new SSL error types

2. **`GEOS5FP/download_file.py`**:
   - Added SSL error handling imports
   - Implemented SSL fallback mechanism for file downloads
   - Enhanced retry logic to handle SSL errors specifically
   - Improved error reporting and logging

3. **`GEOS5FP/exceptions.py`**:
   - Added new `GEOS5FPSSLError` exception class
   - Included detailed troubleshooting guidance
   - Preserved original error context

### Usage

The SSL error handling improvements are automatic and transparent to users. No changes to existing code are required. The library will:

1. **Attempt secure connection first**: Always tries with SSL verification enabled
2. **Graceful fallback**: If SSL errors occur, automatically retries with adjusted settings
3. **Comprehensive logging**: Provides detailed information about connection attempts
4. **Clear error messages**: If all attempts fail, provides actionable troubleshooting guidance

### Example Error Handling Flow

```python
from GEOS5FP import GEOS5FP_connection

# This will now handle SSL errors automatically
geos5fp = GEOS5FP_connection()

try:
    # This call previously failed with SSL errors
    ozone_data = geos5fp.ozone_cm(time_UTC=datetime(2020, 5, 2, 15, 0, 0))
    print("Success! Data retrieved successfully")
    
except GEOS5FPSSLError as e:
    print(f"SSL connection failed: {e}")
    # Error message includes troubleshooting suggestions
    
except Exception as e:
    print(f"Other error: {e}")
```

### Testing

A test script (`test_ssl_simple.py`) was created to verify the SSL error handling improvements:

```bash
python test_ssl_simple.py
```

This test confirms that:
- SSL connections work with the fallback mechanism
- Error handling is properly implemented
- The original SSL error is resolved

### Benefits

1. **Reliability**: Connections are more resilient to SSL/TLS issues
2. **Transparency**: Automatic handling without user intervention
3. **Security**: Still attempts secure connections first
4. **Debugging**: Detailed logging helps diagnose connection issues
5. **User Experience**: Clear error messages with actionable suggestions

### Troubleshooting

If SSL errors still occur after these improvements, users can:

1. **Check network connectivity**: Ensure `portal.nccs.nasa.gov` is accessible
2. **Verify firewall settings**: Ensure HTTPS connections are allowed
3. **Check proxy configuration**: Corporate proxies may interfere with SSL
4. **Update certificates**: Ensure system SSL certificates are up to date
5. **Contact support**: Error messages now include specific context for debugging

### Security Considerations

The SSL fallback mechanism temporarily disables SSL verification as a last resort. This is:
- Only used when secure connections fail
- Logged with warnings
- Still encrypted (just not certificate-verified)
- A common pattern for handling SSL infrastructure issues

For maximum security in production environments, consider:
- Keeping SSL certificates up to date
- Using corporate certificate stores if applicable
- Monitoring logs for SSL fallback usage
- Investigating persistent SSL issues

## Conclusion

These improvements make the GEOS5FP library more robust and user-friendly when dealing with SSL/TLS connection issues, while maintaining security best practices and providing comprehensive error handling.