"""
Risk Measures: Distance-to-Default and Default Probability

Compute credit risk measures from calibrated asset parameters.
"""

import numpy as np
from scipy.stats import norm


def distance_to_default(V, D, T, r, sigma_V):
    """
    Calculate distance-to-default (DD).
    
    DD measures how many standard deviations the asset value is above
    the default threshold.
    
    Parameters:
    -----------
    V : float
        Current asset value
    D : float
        Face value of debt
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualized)
    sigma_V : float
        Asset volatility (annualized)
    
    Returns:
    --------
    float
        Distance-to-default
    """
    # DD = (E[V_T] - D) / std(V_T)
    # E[V_T] = V * exp(r*T)
    # std(V_T) = V * exp(r*T) * sqrt(exp(sigma_V^2*T) - 1)
    
    expected_asset_value = V * np.exp(r * T)
    std_asset_value = expected_asset_value * np.sqrt(np.exp(sigma_V**2 * T) - 1)
    
    # Avoid division by zero
    if std_asset_value < 1e-8:
        return np.inf if expected_asset_value > D else -np.inf
        
    return (expected_asset_value - D) / std_asset_value


def default_probability(V, D, T, r, sigma_V):
    """
    Calculate risk-neutral default probability (PD).
    
    The probability that asset value at maturity will be below debt.
    
    Parameters:
    -----------
    V : float
        Current asset value
    D : float
        Face value of debt
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualized)
    sigma_V : float
        Asset volatility (annualized)
    
    Returns:
    --------
    float
        Default probability (between 0 and 1)
    """
    # PD = Phi(-d2) where d2 = (ln(V/D) + (r - sigma_V^2/2)*T) / (sigma_V*sqrt(T))
    
    numerator = np.log(V / D) + (r - 0.5 * sigma_V**2) * T
    denominator = sigma_V * np.sqrt(T)
    
    d2 = numerator / denominator
    
    return norm.cdf(-d2)


def compute_risk_measures(V, D, T, r, sigma_V):
    """
    Compute both distance-to-default and default probability.
    
    Parameters:
    -----------
    V : float
        Current asset value
    D : float
        Face value of debt
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualized)
    sigma_V : float
        Asset volatility (annualized)
    
    Returns:
    --------
    dict
        Dictionary with 'DD' and 'PD' keys
    """
    DD = distance_to_default(V, D, T, r, sigma_V)
    PD = default_probability(V, D, T, r, sigma_V)
    
    return {
        'DD': DD,
        'PD': PD
    }
