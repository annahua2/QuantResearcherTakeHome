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
    
    This solves the system of equations:
    1. E = BlackScholes(V, D, T, r, sigma_V)
    2. sigma_E * E = (∂E/∂V) * sigma_V * V
    
    where ∂E/∂V = Φ(d₁) is the option delta.
    
    Parameters:
    -----------
    E : float
        Market value of equity
    sigma_E : float
        Equity volatility (annualized)
    D : float
        Face value of debt
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualized)
    V0 : float, optional
        Initial guess for asset value (default: E + D)
    sigma_V0 : float, optional
        Initial guess for asset volatility (default: sigma_E * E / (E + D))
    
    Returns:
    --------
    tuple (V, sigma_V)
        Estimated asset value and asset volatility
    """
    
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
        
        # Enforce positive values for numerical stability
        V = np.abs(V)
        sigma_V = np.abs(sigma_V)
        
        # Equation 1: Equity value equals call option value
        E_calc = black_scholes_call(V, D, T, r, sigma_V)
        eq1 = E_calc - E
        
        # Equation 2: Equity volatility relationship
        delta = black_scholes_delta(V, D, T, r, sigma_V)
        
        # Avoid division by zero
        if E > 1e-8:
            E_vol_calc = (V / E) * delta * sigma_V
        else:
            E_vol_calc = sigma_V
            
        eq2 = E_vol_calc - sigma_E
        
        return [eq1, eq2]
    
    # Solve the system
    result = fsolve(equations, [V0, sigma_V0], xtol=1e-6)
    V, sigma_V = result
    
    return np.abs(V), np.abs(sigma_V)
