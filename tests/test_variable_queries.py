"""Tests for variable information lookup and queries."""
import pytest
from GEOS5FP import GEOS5FPConnection
from GEOS5FP.constants import GEOS5FP_VARIABLES


@pytest.fixture
def conn():
    """Create a connection for testing."""
    return GEOS5FPConnection()


class TestGetVariableInfo:
    """Tests for _get_variable_info method."""
    
    def test_get_variable_info_exists(self, conn):
        """Test that _get_variable_info method exists."""
        assert hasattr(conn, '_get_variable_info')
        assert callable(conn._get_variable_info)
    
    def test_get_variable_info_sfmc(self, conn):
        """Test looking up SFMC variable."""
        description, product, variable = conn._get_variable_info("SFMC")
        
        assert isinstance(description, str)
        assert isinstance(product, str)
        assert isinstance(variable, str)
        assert product == "tavg1_2d_lnd_Nx"
        assert variable == "SFMC"
    
    def test_get_variable_info_lai(self, conn):
        """Test looking up LAI variable."""
        description, product, variable = conn._get_variable_info("LAI")
        
        assert isinstance(description, str)
        assert product == "tavg1_2d_lnd_Nx"
        assert variable == "LAI"
    
    def test_get_variable_info_ta_k(self, conn):
        """Test looking up Ta_K (air temperature) variable."""
        description, product, variable = conn._get_variable_info("Ta_K")
        
        assert isinstance(description, str)
        assert product == "tavg1_2d_slv_Nx"
        assert variable == "T2M"
    
    def test_get_variable_info_invalid_raises(self, conn):
        """Test that looking up invalid variable raises KeyError."""
        with pytest.raises(KeyError):
            conn._get_variable_info("INVALID_VARIABLE_NAME")
    
    def test_get_variable_info_case_sensitive(self, conn):
        """Test that variable names are case-sensitive."""
        # Should work
        conn._get_variable_info("SFMC")
        
        # Should fail (wrong case)
        with pytest.raises(KeyError):
            conn._get_variable_info("sfmc")


class TestVariableAliases:
    """Tests for variable aliases."""
    
    def test_sm_sfmc_alias(self, conn):
        """Test that SM and SFMC are aliases for the same variable."""
        sm_info = conn._get_variable_info("SM")
        sfmc_info = conn._get_variable_info("SFMC")
        
        assert sm_info == sfmc_info
    
    def test_ta_ta_k_alias(self, conn):
        """Test that Ta and Ta_K are aliases for the same variable."""
        ta_info = conn._get_variable_info("Ta")
        ta_k_info = conn._get_variable_info("Ta_K")
        
        assert ta_info == ta_k_info
    
    def test_ts_ts_k_alias(self, conn):
        """Test that Ts and Ts_K are aliases for the same variable."""
        ts_info = conn._get_variable_info("Ts")
        ts_k_info = conn._get_variable_info("Ts_K")
        
        assert ts_info == ts_k_info
    
    def test_vapor_alias(self, conn):
        """Test that vapor_kgsqm and vapor_gccm are aliases."""
        vapor_kgsqm = conn._get_variable_info("vapor_kgsqm")
        vapor_gccm = conn._get_variable_info("vapor_gccm")
        
        assert vapor_kgsqm == vapor_gccm
    
    def test_ozone_alias(self, conn):
        """Test that ozone_dobson and ozone_cm are aliases."""
        ozone_dobson = conn._get_variable_info("ozone_dobson")
        ozone_cm = conn._get_variable_info("ozone_cm")
        
        assert ozone_dobson == ozone_cm


class TestVariableMethods:
    """Tests for variable-specific methods."""
    
    def test_lai_method_exists(self, conn):
        """Test that LAI method exists."""
        assert hasattr(conn, 'LAI')
        assert callable(conn.LAI)
    
    def test_sfmc_method_exists(self, conn):
        """Test that SFMC method exists."""
        assert hasattr(conn, 'SFMC')
        assert callable(conn.SFMC)
    
    def test_sm_method_exists(self, conn):
        """Test that SM method exists."""
        assert hasattr(conn, 'SM')
        assert callable(conn.SM)
    
    def test_ta_k_method_exists(self, conn):
        """Test that Ta_K method exists."""
        assert hasattr(conn, 'Ta_K')
        assert callable(conn.Ta_K)
    
    def test_ts_k_method_exists(self, conn):
        """Test that Ts_K method exists."""
        assert hasattr(conn, 'Ts_K')
        assert callable(conn.Ts_K)
    
    def test_variable_method_exists(self, conn):
        """Test that generic variable method exists."""
        assert hasattr(conn, 'variable')
        assert callable(conn.variable)


class TestAllVariablesAccessible:
    """Test that all variables in GEOS5FP_VARIABLES are accessible."""
    
    def test_all_variables_have_info(self, conn):
        """Test that all variables can be looked up."""
        for var_name in GEOS5FP_VARIABLES.keys():
            info = conn._get_variable_info(var_name)
            assert info is not None
            assert len(info) == 3
    
    def test_all_variables_have_valid_structure(self, conn):
        """Test that all variables have valid info structure."""
        for var_name in GEOS5FP_VARIABLES.keys():
            description, product, variable = conn._get_variable_info(var_name)
            
            assert isinstance(description, str)
            assert len(description) > 0
            assert isinstance(product, str)
            assert len(product) > 0
            assert isinstance(variable, str)
            assert len(variable) > 0
