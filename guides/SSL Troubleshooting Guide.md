# SSL and Download Troubleshooting

## Current download behavior

`GEOS5FPConnection.download_file()` uses Pooch's HTTP downloader for streamed
GET requests and requests' normal certificate verification. Its pre-download
HEAD checks also use a retry-configured session with certificate verification
enabled. The GEOS-5 FP download path does not disable certificate checks or
weaken TLS settings as a fallback.

Older versions and experiments used legacy SSL fallback helpers. Those helpers
are not used by the current `GEOS5FPConnection.download_file()` workflow, so
older logs or guides describing automatic SSL downgrades do not describe the
current download behavior.

## Staging and retries

The configured `download_directory` is the final location for downloaded
granules. It defaults to `~/data/GEOS5FP`. During each attempt, network chunks
are written to a fresh temporary directory under
`pooch.os_cache("GEOS5FP")`, outside the configured download directory. The
temporary staging directory is removed after the attempt.

After the transfer completes, the file is published to the destination and
validated as a GEOS-5 FP NetCDF file. Invalid files are removed and retried.
When staging and destination are on different filesystems, the completed
download is copied to a temporary file beside the destination and then
atomically renamed; a sync client may observe this brief publication file, but
the network transfer is not streamed into the synchronized destination.

The pre-download HEAD checks use the connection's retry-configured session.
Transfer and HTTP failures are retried according to the `retries` and
`wait_seconds` arguments. SSL errors are reported as `GEOS5FPSSLError`; other
download failures are reported as `FailedGEOS5FPDownload`. Missing days and
missing granules have their own exception types.

```python
from GEOS5FP import GEOS5FPConnection

connection = GEOS5FPConnection(download_directory="~/data/GEOS5FP")
# Set granule_url to the URL of the requested GEOS-5 FP granule.
granule = connection.download_file(
    URL=granule_url,
    retries=3,
    wait_seconds=30,
)
print(granule.filename)
```

## Diagnosing connection problems

### 1. Check the GEOS-5 FP endpoint with certificate verification enabled

```bash
curl -I --connect-timeout 15 --max-time 30 \
  "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/"
```

Do not use `curl -k` or disable certificate verification when diagnosing
production downloads. A successful TLS handshake followed by an HTTP error
(for example, `503`) is a server response, not a certificate-verification
failure.

You can also check Python's normal verified HTTPS path:

```python
import requests

url = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/"
response = requests.head(url, timeout=30)
print(response.status_code)
```

### 2. Check proxy and firewall configuration

Corporate proxies and firewalls can interrupt TLS connections. Review the
configured proxy settings and verify that HTTPS traffic to
`portal.nccs.nasa.gov` on port 443 is permitted.

```python
import os

print(os.environ.get("HTTP_PROXY"))
print(os.environ.get("HTTPS_PROXY"))
```

### 3. Check the system certificate store

If Python reports a certificate-verification error, update or configure the
system's trusted CA certificates or corporate CA bundle. Do not work around the
problem with `verify=False`, disabled hostname checks, or weaker cipher
settings.

### 4. Enable logging and retry

```python
import logging
from GEOS5FP import GEOS5FPConnection

logging.basicConfig(level=logging.INFO)
connection = GEOS5FPConnection()
granule = connection.download_file(
    URL=granule_url,
    retries=3,
    wait_seconds=30,
)
```

If the failure persists, capture the exception type and message, the HTTP
status (if any), Python/OpenSSL versions, and whether a proxy is in use. Avoid
sharing credentials or other sensitive proxy configuration.
