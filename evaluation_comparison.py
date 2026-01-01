"""
Comparison script between naive and improved models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create outputs directory if it doesn't exist
Path('report').mkdir(exist_ok=True)


def load_results():
    """Load results from both models."""
    # NOTE: Updated filename to match what we generated earlier (baseline_results.csv)
    naive = pd.read_csv('outputs/baseline_results.csv', parse_dates=['date'])
    improved = pd.read_csv('outputs/improved_results.csv', parse_dates=['date'])
    return naive, improved


def compare_time_series_stability(naive, improved):
    """
    Compare time-series stability of risk measures.
    Lower standard deviation = more stable = better (usually)
    """
    print("\n" + "="*60)
    print("Time-Series Stability Comparison")
    print("="*60)
    
    # Group by firm and compute standard deviation of PD
    naive_stability = naive.groupby('firm_id')['PD'].std()
    improved_stability = improved.groupby('firm_id')['PD'].std()
    
    print("\nPD Standard Deviation (lower is more stable):")
    print(f"\nNaive Model:")
    print(naive_stability)
    print(f"\nImproved Model:")
    print(improved_stability)
    
    print(f"\nAverage PD Std Dev:")
    print(f"  Naive:    {naive_stability.mean():.4f}")
    print(f"  Improved: {improved_stability.mean():.4f}")
    
    return naive_stability, improved_stability


def compare_cross_sectional_ranking(naive, improved):
    """
    Compare cross-sectional risk ranking.
    """
    print("\n" + "="*60)
    print("Cross-Sectional Risk Ranking (Average PD)")
    print("="*60)
    
    # Get average PD per firm
    naive_avg_pd = naive.groupby('firm_id')['PD'].mean().sort_values(ascending=False)
    improved_avg_pd = improved.groupby('firm_id')['PD'].mean().sort_values(ascending=False)
    
    print("\nAverage PD by Firm (sorted, highest to lowest):")
    print(f"\nNaive Model (Values are unreasonably high):")
    print(naive_avg_pd)
    print(f"\nImproved Model (Values are realistic):")
    print(improved_avg_pd)


def plot_comparison(naive, improved, firm_id='AAPL'):
    """
    Plot time series comparison for a specific firm.
    """
    naive_firm = naive[naive['firm_id'] == firm_id].sort_values('date')
    improved_firm = improved[improved['firm_id'] == firm_id].sort_values('date')
    
    # Skip if no data
    if naive_firm.empty or improved_firm.empty:
        print(f"Skipping plot for {firm_id}: No data found.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. PD comparison (Log Scale because Improved PD is very small)
    axes[0, 0].plot(naive_firm['date'], naive_firm['PD'], label='Naive (Unit Mismatch)', color='red', alpha=0.7)
    axes[0, 0].plot(improved_firm['date'], improved_firm['PD'], label='Improved (Market Cap)', color='green', alpha=0.7)
    axes[0, 0].set_title(f'Default Probability (PD): {firm_id}')
    axes[0, 0].set_ylabel('PD (Log Scale)')
    axes[0, 0].set_yscale('log')  # CRITICAL CHANGE: Log scale to see small values
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. DD comparison
    axes[0, 1].plot(naive_firm['date'], naive_firm['DD'], label='Naive', color='red', alpha=0.7)
    axes[0, 1].plot(improved_firm['date'], improved_firm['DD'], label='Improved', color='green', alpha=0.7)
    axes[0, 1].set_title(f'Distance-to-Default (DD): {firm_id}')
    axes[0, 1].set_ylabel('DD')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Asset volatility comparison
    axes[1, 0].plot(naive_firm['date'], naive_firm['sigma_V'], label='Naive', color='red', alpha=0.7)
    axes[1, 0].plot(improved_firm['date'], improved_firm['sigma_V'], label='Improved', color='green', alpha=0.7)
    axes[1, 0].set_title(f'Asset Volatility (sigma_V): {firm_id}')
    axes[1, 0].set_ylabel('Vol')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Asset value comparison
    axes[1, 1].plot(naive_firm['date'], naive_firm['V'], label='Naive', color='red', alpha=0.7)
    axes[1, 1].plot(improved_firm['date'], improved_firm['V'], label='Improved', color='green', alpha=0.7)
    axes[1, 1].set_title(f'Asset Value (V): {firm_id}')
    axes[1, 1].set_ylabel('Value ($)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to 'report' folder instead of 'outputs'
    plot_path = f'report/comparison_{firm_id}.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot to {plot_path}")


def main():
    """Main comparison function."""
    print("Model Comparison")
    print("="*60)
    
    # Load results
    try:
        naive, improved = load_results()
        print(f"Loaded {len(naive)} baseline records and {len(improved)} improved records.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you've run both models (python -m naive_model AND python -m improved)")
        return
    
    # Compare metrics
    compare_time_series_stability(naive, improved)
    compare_cross_sectional_ranking(naive, improved)
    
    # Plot comparison for ALL firms
    firms = naive['firm_id'].unique()
    print(f"\nGenerating plots for firms: {firms}")
    
    for firm in firms:
        plot_comparison(naive, improved, firm_id=firm)
    
    print("\n" + "="*60)
    print("Comparison complete! Check the 'report/' folder for images.")
    print("="*60)


if __name__ == "__main__":
    main()

