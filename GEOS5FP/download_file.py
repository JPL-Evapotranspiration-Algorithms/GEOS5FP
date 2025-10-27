import os
import sys
import warnings
from datetime import datetime
from time import sleep
import posixpath
from os.path import exists, getsize
from shutil import move
from os import makedirs
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError
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

            # Start timer for download duration
            t = TicToc()
            t.tic()
            logger.info(f"downloading with requests: {URL} -> {expanded_partial_filename}")
            try:
                # Initiate HTTP GET request with streaming
                response = requests.get(URL, stream=True, timeout=120)
                response.raise_for_status()
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
            except (ChunkedEncodingError, ConnectionError) as e:
                # Handle network errors: log, clean up, retry if possible
                logger.error(f"Network error during download: {e}")
                if exists(expanded_partial_filename):
                    os.remove(expanded_partial_filename)
                if retries == 0:
                    raise IOError(f"requests download failed: {URL} -> {partial_filename}")
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
