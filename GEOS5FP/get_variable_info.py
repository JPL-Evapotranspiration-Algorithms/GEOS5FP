"""
Utility for looking up variable metadata from GEOS-5 FP variable definitions.
"""

from typing import Tuple
from GEOS5FP.constants import GEOS5FP_VARIABLES


def get_variable_info(variable_name: str) -> Tuple[str, str, str]:
    """
    Look up variable metadata from constants.
    
    Parameters
    ----------
    variable_name : str
        The name of the variable to look up
    
    Returns
    -------
    tuple of (str, str, str)
        Tuple of (description, product, variable)
    
    Raises
    ------
    KeyError
        If variable_name is not found in GEOS5FP_VARIABLES
    
    Examples
    --------
    >>> description, product, variable = get_variable_info("Ta_K")
    >>> print(f"{description}: {product}.{variable}")
    air temperature at 2 meters in Kelvin: tavg1_2d_slv_Nx.T2M
    """
    if variable_name not in GEOS5FP_VARIABLES:
        raise KeyError(f"Variable '{variable_name}' not found in GEOS5FP_VARIABLES")
    
    return GEOS5FP_VARIABLES[variable_name]
