# Testing and Test Guidance

## Overview

This guide describes how to run the GEOS5FP test suite and what the download
and validation tests cover. Test results depend on the checkout and environment;
this guide does not claim a fixed suite count or runtime.

## Test Scope

The suite includes unit tests for connection setup, GEOS-5 FP file validation,
and data retrieval behavior. The download transport tests use mocked HTTP
responses, so they do not depend on a live NASA server or large granule
downloads.

### Download Transport Tests

`tests/test_connection_download.py` covers:

- Pooch's streamed HTTP downloader writing chunks to temporary staging outside
  the configured destination directory
- Verified HTTPS defaults for the HEAD and GET requests
- Retry behavior after a transient network failure
- Publication of a completed download, including a simulated cross-filesystem
  move
- Missing-day and missing-granule exception mapping

These tests exercise the package's public `GEOS5FPConnection.download_file()`
method. They do not test legacy SSL fallback behavior.

### Run Tests

Run the complete suite from the repository root:

```bash
python -m pytest
```

Run the focused download and connection tests:

```bash
python -m pytest tests/test_connection_download.py tests/test_connection.py
```

Run with verbose output or coverage:

```bash
python -m pytest -v
python -m pytest --cov=GEOS5FP
```

## Validation System Tests

The validation system includes comprehensive tests covering:

- ✅ File existence and accessibility checks
- ✅ File size validation (min/max thresholds)
- ✅ Filename pattern matching
- ✅ NetCDF format validation
- ✅ Spatial reference system checks
- ✅ Variable and subdataset validation
- ✅ Data integrity sampling
- ✅ Directory batch validation
- ✅ Error handling for various failure scenarios

The validation tests check that the validator identifies valid and invalid
files and reports the associated results.

## Pytest Best Practices

Pytest test functions should use assertions rather than return `True` or
`False`. Use `pytest.fail()` for explicit failures and `pytest.skip()` when a
test's prerequisites are unavailable.

### Assertion Patterns

Here are the recommended patterns for different test scenarios:

#### 1. Successful Operations

```python
def test_successful_operation():
    result = perform_operation()
    assert result is not None, "Result should not be None"
    assert isinstance(result, ExpectedType), f"Expected type {ExpectedType}, got {type(result)}"
```

#### 2. HTTP Status Validation

```python
def test_http_request():
    response = make_request(url)
    assert response.status_code in [200, 404], f"Unexpected status code: {response.status_code}"
```

#### 3. Expected Errors

```python
import pytest

def test_missing_day():
    from GEOS5FP.exceptions import GEOS5FPDayNotAvailable

    with pytest.raises(GEOS5FPDayNotAvailable):
        connection.download_file(URL=url)
```

#### 4. Explicit Failures

```python
def test_critical_functionality():
    result = critical_operation()
    if result is None:
        pytest.fail("Critical operation failed - result is None")
    # Continue with more checks...
```

#### 5. Conditional Skips

```python
def test_optional_feature():
    try:
        import optional_dependency
    except ImportError as e:
        pytest.skip(f"Optional dependency not available: {e}")
    
    # Test code using optional_dependency...
```

### Benefits of Proper Pytest Usage

1. **Clean Test Output** - Avoid warnings about returned values
2. **Better Error Messages** - More descriptive assertion messages
3. **Proper Test Semantics** - Follow pytest best practices
4. **Maintainability** - Make test intent and failures easier to understand

## Writing New Tests

### Template for New Test Functions

```python
import pytest
from GEOS5FP import GEOS5FP

def test_new_feature():
    """
    Test description: what this test verifies.
    """
    # Setup
    geos5fp = GEOS5FP()
    test_input = "some_value"
    
    try:
        # Execute
        result = geos5fp.some_method(test_input)
        
        # Assert
        assert result is not None, "Result should not be None"
        assert isinstance(result, ExpectedType), "Result should be expected type"
        assert result.some_property == expected_value, "Property should match expected value"
        
        # If test reaches here, it passed (returns None implicitly)
        
    except ExpectedError as e:
        # Handle expected errors
        pytest.skip(f"Expected condition not met: {e}")
        
    except UnexpectedError as e:
        # Fail on unexpected errors
        pytest.fail(f"Unexpected error occurred: {e}")
```

### Test Organization

```python
# Group related assertions
def test_validation_comprehensive():
    """Test comprehensive validation checks."""
    result = validate_file("test.nc4")
    
    # File validation
    assert result is not None
    assert hasattr(result, 'is_valid')
    assert hasattr(result, 'errors')
    
    # Validity checks
    if result.is_valid:
        assert len(result.errors) == 0, "Valid files should have no errors"
        assert result.metadata is not None, "Valid files should have metadata"
    else:
        assert len(result.errors) > 0, "Invalid files should have errors"
```

### Using Fixtures

```python
import pytest
from GEOS5FP import GEOS5FP

@pytest.fixture
def geos5fp_connection():
    """Fixture providing GEOS5FP connection instance."""
    return GEOS5FP()

@pytest.fixture
def test_coordinates():
    """Fixture providing test coordinate data."""
    return {
        'lat': 35.799,
        'lon': -76.656
    }

def test_with_fixtures(geos5fp_connection, test_coordinates):
    """Test using fixtures."""
    result = geos5fp_connection.Ta_K(
        time_UTC="2020-05-02 15:00:00",
        lat=test_coordinates['lat'],
        lon=test_coordinates['lon']
    )
    assert result is not None
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=GEOS5FP

# Run specific test file
python -m pytest tests/test_connection_download.py

# Run specific test function
python -m pytest tests/test_connection_download.py::test_download_uses_pooch_staging_and_retries

# Run with verbose output
python -m pytest -v

# Run with detailed output including print statements
python -m pytest -v -s
```

### Test Filtering

```bash
# Run tests matching pattern
python -m pytest -k "download"

# Run tests in specific directory
python -m pytest tests/

# Run only failed tests from last run
python -m pytest --lf

# Run failed tests first, then others
python -m pytest --ff
```

### Continuous Integration

The test suite is designed to run in CI/CD environments:

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run tests
        run: python -m pytest --cov=GEOS5FP
```

## Test Coverage

The test suite covers:
- Core data retrieval methods
- Download staging, retries, verified HTTPS defaults, and download exceptions
- File validation and integrity checks
- Exception handling
- Package imports and connection behavior

## Troubleshooting Test Failures

### Common Issues

#### 1. Network Connectivity
```
HTTPSConnectionPool(host='portal.nccs.nasa.gov', port=443): Read timed out
```

**Solution**: Check internet access, proxy configuration, and NASA portal
availability. The focused download transport tests use mocked responses and do
not require the live portal.

#### 2. Missing Dependencies
```
ImportError: No module named 'rasterio'
```

**Solution**: Install required dependencies:
```bash
pip install -e ".[dev]"
```

#### 3. SSL Certificate Issues
```
requests.exceptions.SSLError: certificate verify failed
```

**Solution**: Check the system or corporate CA certificates, proxy
configuration, and firewall. Do not disable certificate verification; see the
[SSL Troubleshooting Guide](./SSL%20Troubleshooting%20Guide.md).

## Summary

The GEOS5FP test suite:

1. Covers core functionality, downloads, and NetCDF validation
2. Tests the downloader with mocked HTTP responses
3. Checks HTTPS verification defaults and GEOS5FP-specific exceptions
4. Can be run locally or in CI with `python -m pytest`
