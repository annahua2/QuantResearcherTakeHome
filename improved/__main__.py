"""
Entry point for improved Merton model.

Run with: python -m improved
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from improved.model import MertonModel
from improved.calibration import calibrate_asset_parameters
from improved.risk_measures import compute_risk_measures


def get_shares_outstanding():
    return {
        'AAPL': 17350.0,
        'JPM': 3050.0,
        'TSLA': 960.0,
        'XOM': 4270.0,
        'F': 3970.0
    }


def main():
    print("IMPROVED Merton Model")
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

    equity_prices['year'] = equity_prices['date'].dt.year
    debt['year'] = debt['date'].dt.year

    df = pd.merge(equity_prices, equity_vol, on=['date', 'firm_id'], how='inner')
    df = pd.merge(df, risk_free, on='date', how='left')
    
    debt_annual = debt[['firm_id', 'year', 'debt']].drop_duplicates()
    df = pd.merge(df, debt_annual, on=['firm_id', 'year'], how='left')
    
    df = df.dropna(subset=['equity_price', 'equity_vol', 'debt', 'risk_free_rate'])

    ## improve: Fix unit mismatch by calculating Market Cap (Price * Shares)
    shares_map = get_shares_outstanding()
    df['shares'] = df['firm_id'].map(shares_map)
    df['market_equity'] = df['equity_price'] * df['shares']
    
    df = df.dropna(subset=['market_equity'])

    print(f"Processing {len(df)} records...")
    
    results = []
    T = 1.0
    
    for row in df.itertuples():
        ## improve: Use market_equity instead of share price
        E = row.market_equity 
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
            'share_price': row.equity_price,
            'market_cap': E,
            'V': V,
            'sigma_V': sigma_V,
            'DD': risk['DD'],
            'PD': risk['PD']
        })
    
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    results_df = pd.DataFrame(results)
    output_file = output_dir / 'improved_results.csv'
    results_df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to {output_file}")
    
    if not results_df.empty:
        print("\nAverage PD by Firm (Improved):")
        print(results_df.groupby('firm_id')['PD'].mean())


if __name__ == "__main__":
    main()
