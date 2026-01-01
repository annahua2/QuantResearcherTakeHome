"""
Asset Value and Volatility Calibration

Calibrate unobservable asset value (V) and asset volatility (sigma_V)
from observable equity value (E) and equity volatility (sigma_E).
"""

import numpy as np
from scipy.optimize import fsolve

from naive_model.model import black_scholes_call, black_scholes_delta


def calibrate_asset_parameters(E, sigma_E, D, T, r, V0=None, sigma_V0=None):
    """
    Calibrate asset value (V) and asset volatility (sigma_V) from equity data.
    """
    
    if E <= 0 or sigma_E <= 0 or D <= 0 or T <= 0:
        return np.nan, np.nan

    if V0 is None:
        V0 = E + D
    if sigma_V0 is None:
        sigma_V0 = sigma_E * E / (E + D) if (E + D) > 0 else sigma_E
    
    def equations(params):
        """
        System of equations to solve.
        
        Returns:
        --------
        list [eq1, eq2]
            Residuals that should be zero at solution
        """
        V, sigma_V = params
        
        if V <= 0 or sigma_V <= 0:
            return [1e6, 1e6]
        
        E_calc = black_scholes_call(V, D, T, r, sigma_V)
        eq1 = E_calc - E
        
        delta = black_scholes_delta(V, D, T, r, sigma_V)
        E_vol_calc = (delta * sigma_V * V) / E
        eq2 = E_vol_calc - sigma_E
        
        return [eq1, eq2]
    
    result, info, ier, msg = fsolve(
        equations,
        [V0, sigma_V0],
        xtol=1e-6,
        full_output=True
    )
    V, sigma_V = result
    
    if ier != 1 or V <= 0 or sigma_V <= 0:
        return np.nan, np.nan
    
    return V, sigma_V
