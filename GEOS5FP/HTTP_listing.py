import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3 import Retry

from .constants import DEFAULT_READ_TIMEOUT, DEFAULT_RETRIES


def HTTP_listing(
        url: str,
        timeout: float = None,
        retries: int = None,
        username: str = None,
        password: str = None,
        **kwargs):
    """
    Get the directory listing from an FTP-like HTTP data dissemination system.
    There is no standard for listing directories over HTTP, and this was designed
    for use with the USGS data dissemination system.
    HTTP connections are typically made for brief, single-use periods of time.
    :param url: URL of URL HTTP directory
    :param timeout:
    :param retries:
    :param username: username string (optional)
    :param password: password string (optional)
    :param kwargs:
    :return:
    """
    if timeout is None:
        timeout = DEFAULT_READ_TIMEOUT

    if retries is None:
        retries = DEFAULT_RETRIES

    retries = Retry(
        total=retries,
        backoff_factor=3,
        status_forcelist=[500, 502, 503, 504]
    )

    if not username is None and not password is None:
        auth = HTTPBasicAuth(username, password)
    else:
        auth = None

    with requests.Session() as s:
        # too many retries in too short a time may cause the server to refuse connections
        s.mount('http://', HTTPAdapter(max_retries=retries))
        response = s.get(
            url,
            auth=auth,
            timeout=timeout
        )

    # there was a conflict between Unicode markup and from_encoding
    soup = BeautifulSoup(response.text, 'html.parser')
    links = list(soup.find_all('a', href=True))

    # get directory names from links on http site
    directories = [link['href'].replace('/', '') for link in links]

    return directories
