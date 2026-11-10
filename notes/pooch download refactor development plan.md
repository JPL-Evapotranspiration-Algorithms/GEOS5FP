# GEOS5FP download refactor development plan

## Goal

Replace the package's hand-written `GEOS5FP/download_file.py` transport layer with
[`pooch`](https://www.fatiando.org/pooch/latest/), remove that submodule, and
preserve the existing GEOS-5 FP download behavior exposed by
`GEOS5FPConnection.download_file`.

The download staging location must be outside OneDrive/Google Drive-backed
download directories. Pooch streams data in chunks to a temporary file before
publishing it, but its temporary file is normally created in the Pooch cache
directory. Therefore, placing the Pooch cache in the synchronized destination
directory would not meet this requirement.

## Design decisions

- Keep `GEOS5FPConnection.download_file` as the public package-level operation;
  remove only the standalone `GEOS5FP.download_file` module and its direct
  imports.
- Use a fresh `TemporaryDirectory` for each Pooch retrieval, outside the
  configured `download_directory`. Pass it as Pooch's cache path with
  `known_hash=None`, then move the completed file to its final destination and
  run the existing NetCDF validation. A per-download directory avoids stale
  cache entries when no authoritative checksum is available and keeps Pooch's
  in-progress file out of synchronized folders. Clean up staging files on
  success and failure.
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

1. **Confirm Pooch behavior and dependency**
   - Add `pooch` to the project dependencies.
   - Use Pooch's streamed HTTP downloader and temporary-file handling.
   - Choose a staging-directory lifecycle that avoids stale cache entries when
     no authoritative checksum is available, and ensure the staging directory
     is not under `download_directory`.
2. **Integrate with the connection workflow**
   - Replace the standalone downloader call in
     `GEOS5FP/GEOS5FP_connection.py`.
   - Preserve remote-existence checks and map HTTP/network errors into the
     package's existing exception types.
   - Keep GEOS-5 FP NetCDF validation and the outer retry loop authoritative;
     avoid accidentally multiplying Pooch retries by connection retries.
   - Publish only a completed staged file to the requested destination and
     clean up on all error paths.
3. **Remove the standalone module and update callers**
   - Remove `GEOS5FP/download_file.py` and its import from the connection
     module.
   - Update tests, examples, and utility scripts that import the submodule or
     call its function directly. Retain the connection method API.
4. **Update documentation**
   - Document the Pooch dependency and the staging/destination behavior where
     download configuration is described.
   - Clarify that the configured download directory remains the final
     destination, while in-progress transfer files are staged elsewhere.

## Validation

- Unit-test chunked transfer, staging outside the destination directory,
  successful publication, cleanup after transfer errors, and retry behavior
  using mocked HTTP responses (no large NASA downloads).
- Test that a valid existing granule is reused and an invalid existing or
  newly downloaded granule follows the existing validation/cleanup behavior.
- Test missing-file and missing-day exception mapping, plus transient HTTP
  failure handling.
- Run the focused downloader/connection tests, then the full test suite and
  package build.
- Confirm no remaining package, test, example, or script imports
  `GEOS5FP.download_file`.

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