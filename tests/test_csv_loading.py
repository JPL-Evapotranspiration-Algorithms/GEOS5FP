"""Test that variables.csv file is properly loaded and accessible."""
import os
import csv
from pathlib import Path


def test_variables_csv_exists():
    """Test that variables.csv file exists in the package."""
    from GEOS5FP import constants
    
    variables_csv = Path(constants.__file__).parent / "variables.csv"
    assert variables_csv.exists(), f"variables.csv not found at {variables_csv}"


def test_csv_format():
    """Test that variables.csv has the correct format."""
    from GEOS5FP import constants
    
    variables_csv = Path(constants.__file__).parent / "variables.csv"
    
    with open(variables_csv, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        
        # Check required columns exist
        assert 'variable_name' in headers
        assert 'description' in headers
        assert 'product' in headers
        assert 'variable' in headers
        
        # Check we have data
        rows = list(reader)
        assert len(rows) > 0, "CSV file is empty"
        
        # Check each row has all required fields
        for row in rows:
            assert row['variable_name'], "Missing variable_name"
            assert row['description'], "Missing description"
            assert row['product'], "Missing product"
            assert row['variable'], "Missing variable"


def test_csv_matches_loaded_constants():
    """Test that loaded GEOS5FP_VARIABLES matches CSV content."""
    from GEOS5FP import constants
    from GEOS5FP.constants import GEOS5FP_VARIABLES
    
    variables_csv = Path(constants.__file__).parent / "variables.csv"
    
    with open(variables_csv, 'r') as f:
        reader = csv.DictReader(f)
        csv_count = sum(1 for _ in reader)
    
    # The loaded constants should have the same number of entries as the CSV
    assert len(GEOS5FP_VARIABLES) == csv_count, \
        f"Mismatch: {len(GEOS5FP_VARIABLES)} loaded variables vs {csv_count} CSV rows"


def test_csv_has_expected_variables():
    """Test that CSV contains all expected variables."""
    from GEOS5FP import constants
    
    variables_csv = Path(constants.__file__).parent / "variables.csv"
    
    expected_variables = {
        "SM", "SFMC", "LAI", "LHLAND", "EFLUX", "PARDR", "PARDF",
        "AOT", "COT", "Ts", "Ts_K", "Ta", "Ta_K", "Tmin", "Tmin_K",
        "PS", "Q", "vapor_kgsqm", "vapor_gccm", "ozone_dobson", "ozone_cm",
        "U2M", "V2M", "CO2SC", "SWin", "SWTDN",
        "ALBVISDR", "ALBVISDF", "ALBNIRDF", "ALBNIRDR", "ALBEDO"
    }
    
    csv_variables = set()
    with open(variables_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_variables.add(row['variable_name'])
    
    missing = expected_variables - csv_variables
    assert not missing, f"Missing variables in CSV: {missing}"


def test_load_variables_function():
    """Test that _load_variables function works correctly."""
    from GEOS5FP.constants import _load_variables
    
    variables = _load_variables()
    
    assert isinstance(variables, dict)
    assert len(variables) > 0
    
    # Test a known variable
    assert "SFMC" in variables
    assert variables["SFMC"] == ("top layer soil moisture", "tavg1_2d_lnd_Nx", "SFMC")
