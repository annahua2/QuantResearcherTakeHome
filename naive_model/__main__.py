"""
Entry point for baseline Merton model.

Run with: python -m basemodel
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from naive_model.model import MertonModel
from naive_model.calibration import calibrate_asset_parameters
from naive_model.risk_measures import compute_risk_measures


def main():
    """
    Main entry point for baseline model.
    
    This function should:
    1. Load data (equity prices, equity vol, debt, risk-free rates)
    2. For each firm and date, calibrate asset value and volatility
    3. Compute risk measures (DD, PD)
    4. Output results (print or save to file)
    """
    print("Baseline Merton Model")
    print("=" * 60)
    
    # Load data from data/real/
    data_dir = Path('data/real')
    
    equity_prices = pd.read_csv(data_dir / 'equity_prices.csv', parse_dates=['date'])
    equity_vol = pd.read_csv(data_dir / 'equity_vol.csv', parse_dates=['date'])
    debt = pd.read_csv(data_dir / 'debt_quarterly.csv', parse_dates=['date'])
    risk_free = pd.read_csv(data_dir / 'risk_free.csv', parse_dates=['date'])
    
    # Data Alignment: Merge and Forward Fill Debt
    df = pd.merge(equity_prices, equity_vol, on=['date', 'firm_id'], how='inner')
    df = pd.merge(df, risk_free, on='date', how='left')
    
    # Merge debt (left join) and then forward fill per firm
    df = pd.merge(df, debt, on=['date', 'firm_id'], how='left')
    df = df.sort_values(['firm_id', 'date'])
    df['debt'] = df.groupby('firm_id')['debt'].ffill()
    
    # Drop rows where data is missing (e.g. before first debt report)
    df = df.dropna(subset=['equity_price', 'equity_vol', 'debt', 'risk_free_rate'])

    # For each firm and date:
    results = []
    T = 1.0
    
    print(f"Processing {len(df)} records...")
    
    for row in df.itertuples():
        # 1. Get equity value (E), equity volatility (sigma_E), debt (D), risk-free rate (r)
        E = row.equity_price
        sigma_E = row.equity_vol
        D = row.debt
        r = row.risk_free_rate
        
        # 3. Calibrate: V, sigma_V = calibrate_asset_parameters(E, sigma_E, D, T, r)
        V, sigma_V = calibrate_asset_parameters(E, sigma_E, D, T, r)
        
        # Check for calibration failure
        if np.isnan(V) or np.isnan(sigma_V):
            continue
            
        # 4. Compute: DD, PD = compute_risk_measures(V, D, T, r, sigma_V)
        risk = compute_risk_measures(V, D, T, r, sigma_V)
        
        # 5. Store results
        results.append({
            'date': row.date,
            'firm_id': row.firm_id,
            'V': V,
            'sigma_V': sigma_V,
            'DD': risk['DD'],
            'PD': risk['PD']
        })
    
    # Output results
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    results_df = pd.DataFrame(results)
    output_file = output_dir / 'baseline_results.csv'
    results_df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
