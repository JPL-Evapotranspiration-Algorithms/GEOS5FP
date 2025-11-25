"""
Utility for calculating relative humidity from specific humidity, pressure, and temperature.
"""

import numpy as np


def calculate_RH(Q: np.ndarray, PS: np.ndarray, Ta_K: np.ndarray) -> np.ndarray:
    """
    Calculate relative humidity from specific humidity, surface pressure, and air temperature.
    
    Uses the Magnus formula for saturated vapor pressure and thermodynamic relationships
    to compute relative humidity from specific humidity.
    
    Parameters
    ----------
    Q : np.ndarray
        Specific humidity (kg/kg) - mass of water vapor per mass of moist air
    PS : np.ndarray
        Surface pressure (Pa)
    Ta_K : np.ndarray
        Air temperature (K)
    
    Returns
    -------
    np.ndarray
        Relative humidity (0-1), clipped to valid range
    
    Notes
    -----
    The calculation follows these steps:
    1. Convert temperature from Kelvin to Celsius
    2. Calculate saturated vapor pressure using Magnus formula:
       SVP = 611.2 * exp((17.67 * T_C) / (T_C + 243.5))
    3. Convert specific humidity to mixing ratio: w = Q / (1 - Q)
    4. Calculate saturated mixing ratio: ws = ε * SVP / (PS - SVP)
       where ε = Mw / (Md * 1000) ≈ 0.622
    5. Calculate relative humidity: RH = w / ws
    6. Clip to valid range [0, 1]
    
    References
    ----------
    - Magnus formula: Alduchov & Eskridge (1996)
    - Thermodynamic relationships: Wallace & Hobbs (2006)
    """
    # Convert temperature to Celsius
    Ta_C = Ta_K - 273.15
    
    # Calculate saturated vapor pressure using Magnus formula (Pa)
    SVP_Pa = 611.2 * np.exp((17.67 * Ta_C) / (Ta_C + 243.5))
    
    # Calculate relative humidity from specific humidity
    # Molecular weights
    Mw = 18.015268  # g/mol - water vapor
    Md = 28.96546e-3  # kg/mol - dry air
    epsilon = Mw / (Md * 1000)  # dimensionless ratio ≈ 0.622
    
    # Convert specific humidity to mixing ratio
    w = Q / (1 - Q)
    
    # Calculate saturated mixing ratio
    ws = epsilon * SVP_Pa / (PS - SVP_Pa)
    
    # Calculate relative humidity and clip to valid range
    RH = np.clip(w / ws, 0, 1)
    
    return RH
