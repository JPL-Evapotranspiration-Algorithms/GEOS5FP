# GEOS5FP download refactor development plan

## Goal

Replace the package's hand-written `GEOS5FP/download_file.py` transport layer with
[`pooch`](https://www.fatiando.org/pooch/latest/), remove that submodule, and
preserve the existing GEOS-5 FP download behavior exposed by
`GEOS5FPConnection.download_file`.

The download staging location must be outside OneDrive/Google Drive-backed
download directories. The package will direct Pooch's chunked HTTP downloader
to a per-attempt temporary directory under the local GEOS5FP cache, then publish
the completed file to the configured destination.

## Design decisions

- Keep `GEOS5FPConnection.download_file` as the public package-level operation;
  remove only the standalone `GEOS5FP.download_file` module and its direct
  imports.
- Use `pooch.HTTPDownloader` with a fresh `TemporaryDirectory` for each
  attempt, under `pooch.os_cache("GEOS5FP")` and outside the configured
  `download_directory`. This streams chunks without a persistent cache or
  unverified hash scan. Publish the completed file with an atomic rename; if
  staging and destination are on different filesystems, copy to a temporary
  file beside the destination and atomically replace the final path. Clean up
  staging files on success and failure.
- Preserve the existing valid-file short circuit, invalid-file cleanup,
  retry behavior, GEOS-5 FP-specific 404 exceptions, and returned
  `GEOS5FPGranule` behavior.
- Use normal HTTPS certificate verification. Do not retain disabled
  certificate verification or weakened TLS settings in the download path.
  Keep/revise shared SSL fallback helpers only after checking their other
  callers; this refactor should not silently change unrelated HTTP operations.
- GEOS-5 FP downloads do not currently have a trusted checksum registry.
  Continue to rely on the existing product validator rather than treating an
  unverified Pooch hash as a source-integrity check.

## Implementation steps

1. [x] **Confirm Pooch behavior and dependency**
   - [x] Add `pooch` to the project dependencies.
   - [x] Use Pooch's streamed HTTP downloader with temporary-file staging.
   - [x] Choose a per-attempt staging lifecycle outside `download_directory`;
     avoid persistent cache entries without an authoritative checksum.
2. [x] **Integrate with the connection workflow**
   - [x] Replace the standalone downloader call in
     `GEOS5FP/GEOS5FP_connection.py`.
   - [x] Preserve remote-existence checks and map HTTP/network errors into the
     package's existing exception types.
   - [x] Keep GEOS-5 FP NetCDF validation and the outer retry loop authoritative;
     do not multiply Pooch retries by connection retries.
   - [x] Publish only a completed staged file to the requested destination and
     clean up on all error paths, including cross-filesystem publication.
3. [x] **Remove the standalone module and update callers**
   - [x] Remove `GEOS5FP/download_file.py` and its import from the connection
     module.
   - [x] Update tests, examples, and utility scripts that import the submodule
     or call its function directly. Retain the connection method API.
4. [ ] **Update documentation**
   - [ ] Document the Pooch dependency and the staging/destination behavior
     where download configuration is described.
   - [ ] Clarify that the configured download directory remains the final
     destination, while in-progress transfer files are staged elsewhere.

## Validation

- [x] Unit-test chunked transfer, staging outside the destination directory,
  completed-file publication, cleanup after a transfer error, and retry
  behavior using mocked HTTP responses (no large NASA downloads).
- [ ] Test reuse of a valid existing granule and cleanup/retry after invalid
  existing or newly downloaded granules.
- [x] Test missing-day and missing-granule exception mapping, plus transient
  HTTP failure handling.
- [x] Run focused connection/downloader tests.
- [ ] Run the full test suite and package build.
- [x] Confirm no remaining package, test, example, or script imports
  `GEOS5FP.download_file` after the module is removed.

## Acceptance criteria

- `GEOS5FP/download_file.py` is removed, and the package's download path uses
  Pooch.
- The final destination path and `GEOS5FPConnection.download_file` behavior
  remain compatible.
- In-progress downloads are written outside a synchronized
  `download_directory`; only completed files are published there.
- Existing NetCDF validation and GEOS5FP-specific errors remain effective.
- Downloads use verified HTTPS by default, and focused tests, the full suite,
  and package build pass.