"""Test that variable constants are properly centralized and accessible."""
import pytest
from GEOS5FP.constants import GEOS5FP_VARIABLES
from GEOS5FP.GEOS5FP_connection import GEOS5FPConnection


def test_geos5fp_variables_exist():
    """Test that GEOS5FP_VARIABLES dictionary is properly defined."""
    assert isinstance(GEOS5FP_VARIABLES, dict)
    assert len(GEOS5FP_VARIABLES) > 0


def test_all_expected_variables_present():
    """Test that all expected variables are in the constants."""
    expected_variables = [
        "SM", "SFMC", "LAI", "LHLAND", "EFLUX", "PARDR", "PARDF",
        "AOT", "COT", "Ts", "Ts_K", "Ta", "Ta_K", "Tmin", "Tmin_K",
        "PS", "Q", "vapor_kgsqm", "vapor_gccm", "ozone_dobson", "ozone_cm",
        "U2M", "V2M", "CO2SC", "SWin", "SWTDN",
        "ALBVISDR", "ALBVISDF", "ALBNIRDF", "ALBNIRDR", "ALBEDO"
    ]
    
    for var in expected_variables:
        assert var in GEOS5FP_VARIABLES, f"Variable {var} not found in GEOS5FP_VARIABLES"


def test_variable_tuple_structure():
    """Test that each variable entry has the correct structure."""
    for var_name, var_info in GEOS5FP_VARIABLES.items():
        assert isinstance(var_info, tuple), f"{var_name} should map to a tuple"
        assert len(var_info) == 3, f"{var_name} tuple should have 3 elements (description, product, variable)"
        description, product, variable = var_info
        assert isinstance(description, str), f"{var_name} description should be a string"
        assert isinstance(product, str), f"{var_name} product should be a string"
        assert isinstance(variable, str), f"{var_name} variable should be a string"


def test_get_variable_info_method():
    """Test the _get_variable_info method works correctly."""
    connection = GEOS5FPConnection()
    
    # Test a known variable
    name, product, variable = connection._get_variable_info("SFMC")
    assert name == "top layer soil moisture"
    assert product == "tavg1_2d_lnd_Nx"
    assert variable == "SFMC"
    
    # Test an alias
    name, product, variable = connection._get_variable_info("SM")
    assert name == "top layer soil moisture"
    assert product == "tavg1_2d_lnd_Nx"
    assert variable == "SFMC"


def test_get_variable_info_invalid():
    """Test that _get_variable_info raises error for invalid variable."""
    connection = GEOS5FPConnection()
    
    with pytest.raises(KeyError):
        connection._get_variable_info("INVALID_VARIABLE")


def test_aliases_point_to_same_data():
    """Test that aliases point to the same underlying data."""
    # SM and SFMC should be the same
    assert GEOS5FP_VARIABLES["SM"] == GEOS5FP_VARIABLES["SFMC"]
    
    # Ts and Ts_K should be the same
    assert GEOS5FP_VARIABLES["Ts"] == GEOS5FP_VARIABLES["Ts_K"]
    
    # Ta and Ta_K should be the same
    assert GEOS5FP_VARIABLES["Ta"] == GEOS5FP_VARIABLES["Ta_K"]
    
    # Tmin and Tmin_K should be the same
    assert GEOS5FP_VARIABLES["Tmin"] == GEOS5FP_VARIABLES["Tmin_K"]
    
    # vapor_kgsqm and vapor_gccm should be the same
    assert GEOS5FP_VARIABLES["vapor_kgsqm"] == GEOS5FP_VARIABLES["vapor_gccm"]
    
    # ozone_dobson and ozone_cm should be the same
    assert GEOS5FP_VARIABLES["ozone_dobson"] == GEOS5FP_VARIABLES["ozone_cm"]
