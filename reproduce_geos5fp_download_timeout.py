#!/usr/bin/env python3
"""Reproduce the GEOS5FP streamed-download failure against a chosen directory.

The reported failure occurred while downloading this GEOS-5 FP aerosol granule:

    https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y2026/M10/D03/GEOS.fp.asm.tavg3_2d_aer_Nx.20261003_0730.V01.nc4

The traceback ended at ``f.write(chunk)`` in ``GEOS5FP/download_file.py`` with
``TimeoutError: [Errno 60] Operation timed out``. The original destination was
under ``~/data/...``; on the affected Mac, ``~/data`` resolves into
``~/Library/CloudStorage/OneDrive-JPL/data``. This makes a write/sync issue
plausible, but does not prove it: Chrome may use a different HTTP stack and
save to a different directory.

This script calls GEOS5FP's own ``download_file`` function, so it exercises its
HTTP request, 1 MiB response-chunk loop, and file writes. Run it once with the
original OneDrive destination and again with a local directory outside cloud
storage (for example ``~/tmp/geos5fp-test``). Comparing those runs helps
separate destination-write failures from failures that also occur on local
storage. The output filename is unique so an existing cached granule cannot
cause the downloader to skip the transfer.

Run from the root of a GEOS5FP checkout (or an environment where that checkout
is the imported GEOS5FP package):

    python reproduce_geos5fp_download_timeout.py \
        --destination-dir ~/data/ECOv003_testing/L2T_STARS_SOURCES/GEOS5FP_download/2026.10.03

Then compare with:

    python reproduce_geos5fp_download_timeout.py \
        --destination-dir ~/tmp/geos5fp-test

Successful downloads are left in place for inspection. The default retry
settings match the original invocation; use ``--retries 1 --wait-seconds 0``
for a quicker diagnostic run.
"""

import argparse
import inspect
import logging
import os
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from GEOS5FP.download_file import download_file


GRANULE_URL = (
    "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y2026/M10/D03/"
    "GEOS.fp.asm.tavg3_2d_aer_Nx.20261003_0730.V01.nc4"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise GEOS5FP's streamed downloader against a chosen directory."
    )
    parser.add_argument(
        "--destination-dir",
        type=Path,
        default=Path.home()
        / "data/ECOv003_testing/L2T_STARS_SOURCES/GEOS5FP_download/2026.10.03",
        help=(
            "Directory to write the download into. Defaults to the directory "
            "from the reported failure; pass a local non-cloud directory for "
            "a comparison run."
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="GEOS5FP download attempts (default: 3, matching the reported run).",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=30,
        help="Delay between GEOS5FP retries (default: 30, matching the reported run).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s] %(name)s: %(message)s",
    )

    destination_dir = args.destination_dir.expanduser()
    destination_dir.mkdir(parents=True, exist_ok=True)

    granule_name = Path(urlsplit(GRANULE_URL).path).name
    stem, extension = os.path.splitext(granule_name)
    target_filename = destination_dir / f"{stem}.repro-{uuid4().hex}{extension}"

    print(f"GEOS5FP downloader source: {inspect.getsourcefile(download_file)}")
    print(f"URL: {GRANULE_URL}")
    print(f"Destination: {target_filename}")
    print(f"Resolved destination: {target_filename.resolve()}")

    result = download_file(
        URL=GRANULE_URL,
        filename=str(target_filename),
        retries=args.retries,
        wait_seconds=args.wait_seconds,
    )
    print(f"Download succeeded: {result}")
    print(f"Bytes written: {target_filename.stat().st_size}")


if __name__ == "__main__":
    main()
