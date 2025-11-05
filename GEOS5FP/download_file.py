import os
import ssl
import sys
import warnings
from datetime import datetime
from time import sleep
import posixpath
from os.path import exists, getsize
from shutil import move
from os import makedirs
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, SSLError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.util.ssl_ import create_urllib3_context
from tqdm import tqdm as std_tqdm
import colored_logging as cl
import logging

from pytictoc import TicToc

logger = logging.getLogger(__name__)

def download_file(
    URL: str,
    filename: str,
    retries: int = 3,
    wait_seconds: int = 30
) -> str:
    """
    Downloads a file from a specified URL to a local filename with robust error handling, retry logic, and progress bar.

    Args:
        URL (str): The web address of the file to download.
        filename (str): The local path where the file will be saved.
        retries (int, optional): Number of times to retry the download if it fails. Default is 3.
        wait_seconds (int, optional): Seconds to wait between retries. Default is 30.

    Returns:
        str: The filename of the successfully downloaded file.

    Raises:
        ValueError: If filename is not provided.
        IOError: If the download fails after all retries or if a corrupted file is detected.

    The function will:
        - Check for an existing file and skip download if present and valid.
        - Remove zero-size corrupted files before download.
        - Download the file in chunks, showing a progress bar.
        - Handle network errors and other exceptions, retrying as needed.
        - Clean up partial/corrupted files after failed attempts.
        - Log key events and errors for debugging and monitoring.
    """
    # Validate input filename
    if filename is None:
        raise ValueError("filename must be provided")

    # Expand user directory in filename (e.g., ~ to /home/user)
    expanded_filename = os.path.expanduser(filename)

    # Remove zero-size corrupted file if it exists
    if exists(expanded_filename) and getsize(expanded_filename) == 0:
        logger.warning(f"removing previously created zero-size corrupted file: {filename}")
        os.remove(expanded_filename)

    # If file already exists and is valid, return immediately
    if exists(expanded_filename):
        return filename

    # Attempt download with retry logic
    while retries > 0:
        retries -= 1
        try:
            # Download file from URL
            makedirs(os.path.dirname(expanded_filename), exist_ok=True)
            # Create a temporary filename for partial download (with timestamp)
            partial_filename = f"{filename}.{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.download"
            expanded_partial_filename = os.path.expanduser(partial_filename)

            # Remove zero-size partial file if it exists
            if exists(expanded_partial_filename) and getsize(expanded_partial_filename) == 0:
                logger.warning(f"removing zero-size corrupted file: {partial_filename}")
                os.remove(expanded_partial_filename)

            # Create robust session with SSL error handling
            def create_robust_session(ssl_context=None):
                session = requests.Session()
                retry_strategy = Retry(
                    total=1,
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

            def download_with_ssl_fallback(url, stream=True, timeout=120):
                """Download with SSL error handling and multiple fallback strategies."""
                
                # Strategy 1: Default SSL settings
                try:
                    logger.debug(f"attempting download with default SSL: {url}")
                    session = create_robust_session()
                    response = session.get(url, stream=stream, timeout=timeout, verify=True)
                    response.raise_for_status()
                    return response
                except SSLError as e:
                    logger.warning(f"SSL error during download with default settings: {e}")
                    
                    # Strategy 2: Legacy SSL context
                    try:
                        logger.warning(f"retrying download with legacy SSL context: {url}")
                        legacy_context = create_legacy_ssl_context()
                        session = create_robust_session(ssl_context=legacy_context)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            response = session.get(url, stream=stream, timeout=timeout, verify=False)
                            response.raise_for_status()
                            return response
                    except SSLError as fallback_e:
                        logger.warning(f"Legacy SSL context failed: {fallback_e}")
                        
                        # Strategy 3: Minimal session with no retries
                        try:
                            logger.warning(f"retrying download with minimal SSL configuration: {url}")
                            session = requests.Session()
                            # No retries for this final attempt
                            adapter = HTTPAdapter(max_retries=0)
                            session.mount("https://", adapter)
                            session.mount("http://", adapter)
                            
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                response = session.get(url, stream=stream, timeout=60, verify=False)
                                response.raise_for_status()
                                return response
                        except Exception as final_e:
                            logger.error(f"All SSL download strategies failed. Final error: {final_e}")
                            raise SSLError(f"Failed to download from {url} after trying multiple SSL strategies. Original error: {e}")
                    except Exception as fallback_e:
                        logger.error(f"Download failed with legacy SSL context: {fallback_e}")
                        raise SSLError(f"Failed to download from {url} due to SSL issues. Original error: {e}")
                except Exception as e:
                    logger.error(f"Download request failed: {e}")
                    raise

            # Start timer for download duration
            t = TicToc()
            t.tic()
            logger.info(f"downloading with requests: {URL} -> {expanded_partial_filename}")
            try:
                # Initiate HTTP GET request with streaming and SSL fallback
                response = download_with_ssl_fallback(URL, stream=True, timeout=120)
                # Get total file size from response headers (if available)
                total = int(response.headers.get('content-length', 0))
                # Open temporary file for writing in binary mode
                with open(expanded_partial_filename, 'wb') as f:
                    # Configure tqdm progress bar
                    tqdm_kwargs = dict(
                        desc=posixpath.basename(expanded_partial_filename),
                        total=total,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        leave=True,
                        dynamic_ncols=True,
                        ascii=True,
                        miniters=1,
                        mininterval=0.1,
                        disable=False
                    )
                    # Show progress bar only if stdout is a TTY
                    if sys.stdout.isatty():
                        tqdm_kwargs['file'] = sys.stdout
                    # Download file in 1MB chunks, updating progress bar
                    with std_tqdm(**tqdm_kwargs) as bar:
                        for chunk in response.iter_content(chunk_size=1024*1024):
                            if chunk:
                                f.write(chunk)
                                bar.update(len(chunk))
                                bar.refresh()
            except (ChunkedEncodingError, ConnectionError, SSLError) as e:
                # Handle network and SSL errors: log, clean up, retry if possible
                error_type = "SSL" if isinstance(e, SSLError) else "Network"
                logger.error(f"{error_type} error during download: {e}")
                if exists(expanded_partial_filename):
                    os.remove(expanded_partial_filename)
                if retries == 0:
                    raise IOError(f"requests download failed ({error_type} error): {URL} -> {partial_filename}")
                logger.warning(f"waiting {wait_seconds} seconds for retry")
                sleep(wait_seconds)
                continue
            except Exception as e:
                # Handle other exceptions: log, clean up, abort
                logger.exception(f"Download failed: {e}")
                if exists(expanded_partial_filename):
                    os.remove(expanded_partial_filename)
                raise IOError(f"requests download failed: {URL} -> {partial_filename}")

            # Check if temporary file was created
            if not exists(expanded_partial_filename):
                raise IOError(f"unable to download URL: {URL}")

            # Check for zero-size file after download
            if exists(expanded_partial_filename) and getsize(expanded_partial_filename) == 0:
                logger.warning(f"removing zero-size corrupted file: {partial_filename}")
                os.remove(expanded_partial_filename)
                raise IOError(f"zero-size file from download: {URL} -> {partial_filename}")

            # Move completed file to final destination
            move(expanded_partial_filename, expanded_filename)

            # Log download completion, file size, and elapsed time
            elapsed = t.tocvalue()
            logger.info(f"Download completed: {filename} ({(getsize(expanded_filename) / 1000000):0.2f} MB) ({elapsed:.2f} seconds)")

        except Exception as e:
            # If all retries are exhausted, raise the last exception
            if retries == 0:
                raise e
            # Otherwise, log warning and wait before retrying
            logger.warning(e)
            logger.warning(f"waiting {wait_seconds} seconds for retry")
            sleep(wait_seconds)
            continue
    # Return the filename of the successfully downloaded file
    return filename
