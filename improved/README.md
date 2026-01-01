# Improved Merton Model

This directory contains the improved structural credit model, designed to fix a critical logical error in the baseline implementation.

## Documentation of Improvement

* **Assumption Modified:**
    I changed the input for Equity Value ($E$) from **Share Price** to **Market Capitalization**. The baseline model incorrectly used the price of a single share (e.g., ~$100) as the total equity value of the firm.

* **Justification:**
    This change corrects a fundamental **Unit Mismatch**. The Merton model relies on the identity $V = E + D$. Since the provided Debt data ($D$) is reported as the *Total Debt* (in millions), it is mathematically invalid to add a *per-share* price to it. To make the equation valid, both Equity and Debt must represent firm-wide totals.

* **Addressing the Weakness:**
    The baseline model produced nonsensical results (e.g., 100% default probability for Apple) because it treated companies as having almost zero equity compared to their debt. By using Market Cap ($Price \times Shares$), the improved model correctly accounts for the firm's actual equity cushion, resulting in realistic Default Probabilities (near 0% for healthy firms).

## Implementation Details

The dataset lacked "Shares Outstanding" data, so I implemented a specific fix in `__main__.py`:

1.  Created a `get_shares_outstanding()` function with approximate 2020 share counts for the 5 firms.
2.  Calculated `market_equity = equity_price * shares` to derive the correct total equity value before calibration.

## How to Run

```bash
python -m improved

