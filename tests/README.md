# GEOS5FP Test Suite

## Summary

Comprehensive test suite for the GEOS5FP Python package with **141 passing tests** covering core functionality.

## Test Coverage

### Unit Tests (Fast, No Network)

#### 1. **test_connection.py** (12 tests)
- Connection initialization
- Default and custom parameters
- URL configuration
- Save products settings
- Cache structures

#### 2. **test_geometry.py** (11 tests)
- Point geometry detection
- MultiPoint geometry detection
- Non-point geometry rejection
- Coordinate extraction from points
- Coordinate ordering (lat, lon)
- Empty geometry handling

#### 3. **test_urls.py** (13 tests)
- Year URL formatting
- Month URL formatting
- Day URL formatting
- URL padding with zeros
- Download filename generation
- Date-specific directories

#### 4. **test_variable_queries.py** (20 tests)
- Variable information lookup
- Variable aliases (SM/SFMC, Ta/Ta_K, etc.)
- Method existence checks
- Invalid variable handling
- Case sensitivity
- All variables accessibility

#### 5. **test_variable_constants.py** (6 tests)
- GEOS5FP_VARIABLES dictionary
- Variable tuple structure
- Alias validation
- _get_variable_info method

#### 6. **test_csv_loading.py** (5 tests)
- CSV file existence
- CSV format validation
- CSV-to-constants matching
- Required fields validation
- _load_variables function

#### 7. **test_time_handling.py** (18 tests)
- DateTime parsing
- ISO format handling
- Time range creation
- Date conversions
- Date comparisons
- TimeDelta operations
- UTC handling

#### 8. **test_exceptions.py** (18 tests)
- Exception imports
- Exception raising
- Exception messages
- Exception hierarchy
- Exception catching
- Custom exception types:
  - `GEOS5FPDayNotAvailable`
  - `GEOS5FPGranuleNotAvailable`
  - `FailedGEOS5FPDownload`

#### 9. **test_edge_cases.py** (25 tests)
- Empty geometries
- Invalid coordinates (out of bounds)
- Invalid dates (past/future)
- Leap year handling
- Invalid time ranges
- Invalid variable names
- Null/None values
- Special characters
- Boundary conditions

#### 10. **test_import_dependencies.py** (15 tests)
- All package dependencies:
  - affine, beautifulsoup4, colored_logging
  - geopandas, matplotlib, netcdf4
  - numpy, pandas, pyproj, python-dateutil
  - rasterio, rasters, requests, scipy
  - shapely, urllib3

#### 11. **test_import_geos5fp.py** (1 test)
- Package import validation

## Running Tests

### Run All Tests
```bash
pytest
```

### Run with Verbose Output
```bash
pytest -v
```

### Run Specific Test File
```bash
pytest tests/test_connection.py
```

### Run Specific Test Class
```bash
pytest tests/test_geometry.py::TestIsPointGeometry
```

### Run Specific Test
```bash
pytest tests/test_connection.py::test_connection_initialization
```

### Run with Coverage (if pytest-cov installed)
```bash
pytest --cov=GEOS5FP --cov-report=html
```

## Test Markers

Tests are organized with markers for selective execution:

- `@pytest.mark.unit` - Fast unit tests without network
- `@pytest.mark.integration` - Tests that may require network
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.network` - Tests requiring internet

### Run Only Unit Tests
```bash
pytest -m unit
```

### Run Without Network Tests
```bash
pytest -m "not network"
```

## Configuration

### pytest.ini
- Test discovery patterns
- Minimum pytest version: 6.0
- Test paths: `tests/`
- Markers for test categorization
- Ignore paths for non-test directories

### conftest.py
Shared fixtures:
- `sample_point` - Los Angeles coordinates
- `sample_multipoint` - Multiple US cities
- `sample_datetime` - Test datetime
- `sample_date` - Test date
- `sample_time_range` - 7-day time range
- `known_variables` - List of GEOS-5 FP variables

## Test Statistics

- **Total Tests**: 141
- **Passing**: 141 (100%)
- **Failing**: 0
- **Test Files**: 11
- **Execution Time**: ~1.2 seconds

## Future Test Additions

Potential integration tests (require network/mocking):
- Point query integration tests
- Raster query integration tests
- Time series query tests
- Multi-variable query tests
- Data availability tests
- HTTP listing tests
- Download functionality tests
- Granule data access tests

## Contributing

When adding new features, please:
1. Add corresponding unit tests
2. Ensure all tests pass: `pytest`
3. Maintain test coverage above 80%
4. Follow existing test patterns and naming

## CI/CD Integration

Tests are configured for GitHub Actions with the `.github/workflows/ci.yml` file.

### Local Pre-commit Check
```bash
make test
```

This test suite ensures code quality, prevents regressions, and provides confidence for continuous development.
