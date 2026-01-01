"""
Entry point for baseline Merton model.

Run with: python -m naive_model
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
    print("Baseline Merton Model")
    print("=" * 60)
    
    data_dir = Path('data/real')
    print(f"Loading data from {data_dir}...")

    try:
        equity_prices = pd.read_csv(data_dir / 'equity_prices.csv', parse_dates=['date'])
        equity_vol = pd.read_csv(data_dir / 'equity_vol.csv', parse_dates=['date'])
        debt = pd.read_csv(data_dir / 'debt_quarterly.csv', parse_dates=['date'])
        risk_free = pd.read_csv(data_dir / 'risk_free.csv', parse_dates=['date'])
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Extract year for data alignment
    equity_prices['year'] = equity_prices['date'].dt.year
    debt['year'] = debt['date'].dt.year

    # Merge market data
    df = pd.merge(equity_prices, equity_vol, on=['date', 'firm_id'], how='inner')
    df = pd.merge(df, risk_free, on='date', how='left')

    # Merge debt by year
    debt_annual = debt[['firm_id', 'year', 'debt']].drop_duplicates()
    df = pd.merge(df, debt_annual, on=['firm_id', 'year'], how='left')

    # Drop missing records
    df = df.dropna(subset=['equity_price', 'equity_vol', 'debt', 'risk_free_rate'])

    print(f"Processing {len(df)} records...")

    results = []
    T = 1.0

    for row in df.itertuples():
        E = row.equity_price
        sigma_E = row.equity_vol
        D = row.debt
        r = row.risk_free_rate
        
        V, sigma_V = calibrate_asset_parameters(E, sigma_E, D, T, r)
        
        if np.isnan(V):
            continue
            
        risk = compute_risk_measures(V, D, T, r, sigma_V)
        
        results.append({
            'date': row.date,
            'firm_id': row.firm_id,
            'V': V,
            'sigma_V': sigma_V,
            'DD': risk['DD'],
            'PD': risk['PD']
        })
    
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    results_df = pd.DataFrame(results)
    output_file = output_dir / 'baseline_results.csv'
    results_df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to {output_file}")
    
    if not results_df.empty:
        print("\nAverage PD by Firm:")
        print(results_df.groupby('firm_id')['PD'].mean())


if __name__ == "__main__":
    main()
