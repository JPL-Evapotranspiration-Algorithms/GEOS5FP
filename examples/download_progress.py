"""Download one GEOS-5 FP granule and display the Pooch progress bar.

Run from the repository root:

    python -m examples.download_progress

Each run saves the granule in a new temporary directory so the download is
not skipped because of an existing cached file.
"""

import tempfile
from pathlib import Path

from GEOS5FP.GEOS5FP_connection import GEOS5FPConnection


URL = (
    "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y2026/M10/D03/"
    "GEOS.fp.asm.tavg3_2d_aer_Nx.20261003_0730.V01.nc4"
)


def main():
    download_directory = Path(tempfile.mkdtemp(prefix="geos5fp-progress-"))
    filename = download_directory / URL.rsplit("/", 1)[-1]
    print(f"Downloading {URL}", flush=True)
    print(f"Destination: {filename}", flush=True)

    connection = GEOS5FPConnection(download_directory=str(download_directory))
    connection.download_file(URL=URL, filename=str(filename), retries=1)

    print(f"Downloaded {filename.stat().st_size:,} bytes to {filename}")


if __name__ == "__main__":
    main()
