# Technical Report: Structural Credit Modeling

## 1. Introduction

In this assignment, I implemented and calibrated a structural credit risk model based on the Merton (1974) framework. The goal was to estimate the probability of default (PD) for major US corporations using equity market data and financial statement information.

My approach involved two stages. First, I built a "baseline" model following standard textbook definitions. Upon testing, I diagnosed a critical failure in how the model interpreted the input data, specifically regarding the scale of equity versus debt. Second, I implemented an "improved" model that corrected this unit mismatch. This report details the formulation, the diagnosis of the baseline flaws, and the empirical results of the corrected model, which aligns significantly better with economic reality.

## 2. Model Formulation

### 2.1 Baseline Merton Model

The Merton model treats a firm’s equity as a European call option on its assets ($V$) with a strike price equal to its debt ($D$).

**Key Assumptions:**
* The total value of the firm’s assets $V_t$ follows a Geometric Brownian Motion (GBM).
* Debt consists of a single zero-coupon bond with face value $D$ maturing at time $T$.
* Markets are frictionless, and there are no bankruptcy costs or taxes.
* Default occurs only at maturity $T$ if $V_T < D$.

**Mathematical Formulation:**
The value of equity ($E$) is given by the Black-Scholes call option formula:
$$E = V N(d_1) - D e^{-rT} N(d_2)$$

Where:
$$d_1 = \frac{\ln(V/D) + (r + \frac{1}{2}\sigma_V^2)T}{\sigma_V \sqrt{T}}, \quad d_2 = d_1 - \sigma_V \sqrt{T}$$

**Calibration Approach:**
Since the asset value ($V$) and asset volatility ($\sigma_V$) are not directly observable, they must be inferred from the observable equity value ($E$) and equity volatility ($\sigma_E$). This requires solving a system of two equations, utilizing the relationship between equity delta and volatility:
$$\sigma_E E = \sigma_V V N(d_1)$$

### 2.2 Improved Model

**Assumption Modified:**
I modified the definition of the input variable $E$ (Equity Value). In the baseline model, I used the **Share Price** as the input for $E$. In the improved model, I used **Market Capitalization** (Share Price $\times$ Shares Outstanding).

**Justification:**
This improvement was necessary to satisfy the accounting identity $V = E + D$. The provided debt data represents the *total* debt of the firm (in millions). However, the share price represents only a tiny fraction of the firm's equity. Adding a share price (e.g., \$150) to total debt (e.g., \$100,000,000) creates a massive unit mismatch, mathematically implying the firm is insolvent. To make the equation valid, both Debt and Equity must be expressed as firm-wide totals.

**Mathematical Formulation:**
The core Merton equations remain identical. The only change is the input mapping:
$$E_{input} = P_{share} \times N_{shares}$$

**Calibration Changes:**
The numerical solver remains the same, but the magnitude of $E$ increases by a factor of millions. This shifts the starting point for the solver, allowing it to find a solution where $V > D$, rather than $V \approx D$.

## 3. Calibration Methodology

To solve for the two unknowns ($V$ and $\sigma_V$), I treated the problem as a system of non-linear equations:
1.  $f_1(V, \sigma_V) = \text{BlackScholesCall}(V, D, \dots) - E_{obs} = 0$
2.  $f_2(V, \sigma_V) = N(d_1) V \sigma_V - E_{obs} \sigma_E = 0$

**Numerical Methods:**
I used `scipy.optimize.fsolve` to find the roots of this system.

**Handling Failures:**
* **Initialization:** I initialized the guess for $V$ as $E + D$ and $\sigma_V$ as $\sigma_E \times (E / (E+D))$.
* **Filtering:** Rows containing `NaN` or infinite values in the input data were dropped prior to calibration.
* **Constraints:** If the solver returned a negative asset value (which is economically impossible), those results were discarded.

## 4. Empirical Setup

### 4.1 Implementation Details

* **Time to Maturity ($T$):** I assumed a constant horizon of $T = 1.0$ year for all calculations.
* **Risk-Free Rate:** I used the 10-Year Treasury rate (from FRED data) matched to the specific trading date.
* **Data Alignment (Crucial):**
    The debt data was provided annually (year-end values), while equity data was daily. A standard merge failed because the dates didn't overlap. I implemented a **Year-Based Merge**:
    1.  Extracted the year from both the equity price dates and the debt reporting dates.
    2.  Matched the debt reported for fiscal year $Y$ to every trading day in year $Y$.
    3.  This ensured every daily equity observation had a corresponding debt value.

* **Shares Outstanding:** Since share counts were not in the provided dataset, I hardcoded approximate 2020 share counts for the five target firms (AAPL, JPM, TSLA, XOM, F) to calculate Market Capitalization.

## 5. Baseline Model Diagnosis

### 5.1 Identified Weaknesses

The baseline model exhibited a systematic failure best described as a **Unit Mismatch**.

Because the model used Share Price (~$10-1000) as the total equity value against Total Debt (billions), the calibration logic concluded that $Assets \approx Debt$. In the Merton framework, when assets are barely equal to debt, the firm is "at the money" for default.

**Evidence:**
* **High PDs:** The model predicted Default Probabilities (PD) ranging from 3% to 25% for blue-chip companies.
* **Volatility Transfer:** Because the "equity" (share price) was so small relative to debt, the implied asset volatility ($\sigma_V$) became artificially low to satisfy the equations.

### 5.2 Examples

* **Apple (AAPL):**
    * Input $E$: ~$75 (Share Price)
    * Input $D$: ~$130,000 (Total Debt in millions)
    * **Result:** The model interpreted Apple as having \$130,075 in assets and \$130,000 in debt.
    * **Baseline PD:** ~3.8%. This implies Apple, one of the most cash-rich companies in the world, had a significantly high chance of bankruptcy.

* **Tesla (TSLA):**
    * **Baseline PD:** ~24%. The model viewed Tesla as highly distressed, essentially a coin-flip for survival, solely due to the unit error.

## 6. Improved Model Results

### 6.1 Quantitative Comparison

After correcting the inputs to use Market Capitalization, the results shifted dramatically to realistic levels.

| Firm | Baseline Avg PD | Improved Avg PD | Interpretation |
| :--- | :--- | :--- | :--- |
| **AAPL** | 3.80% | ~0.00% | Apple has a massive equity cushion ($> $1 Trillion). |
| **TSLA** | 23.91% | ~0.00% | Tesla's high market cap protects it from theoretical default. |
| **JPM** | 7.00% | < 0.01% | Typical for a major bank. |
| **F** | 7.26% | ~0.00% | Ford shows low default risk in this period. |

*(Note: "0.00%" represents extremely small values like 1e-10, which effectively means zero risk in this theoretical framework.)*

### 6.2 Why It's Better

The improved model addresses the fundamental weakness by correctly representing the capital structure.

* **Economic Validity:** By using Market Cap (e.g., \$1 Trillion for Apple) vs. Debt (\$130 Billion), the model sees that $V \gg D$.
* **Distance to Default (DD):** The "Distance to Default" calculation now reflects that the firm's asset value must drop by >80% or >90% before hitting the debt barrier. This results in a PD that is near zero, accurately reflecting the market's view that these firms are solvent.

## 7. Limitations

Despite the improvement, the model retains several limitations:

1.  **Debt Structure Simplification:** The model treats all liabilities as a single zero-coupon bond maturing in 1 year. In reality, firms have complex debt structures with varying maturities and coupons.
2.  **Stationary Debt Assumption:** My implementation uses annual debt data applied to every day of the year. This creates a "step function" effect for debt, whereas real debt levels fluctuate continuously.
3.  **Merton's Structural Flaw:** The classic Merton model assumes default can only occur at maturity ($T$). It does not account for "first-passage" default (going bankrupt *before* the year ends due to cash flow issues), which might underestimate risk for highly volatile firms.

## 8. Conclusion

This project highlighted the importance of data validation in financial modeling. The math behind the Black-Scholes and Merton models is robust, but it is sensitive to input definitions.

**Key Takeaways:**
* A naive application of the model led to catastrophic estimation errors (predicting Apple was near-insolvent) due to a unit mismatch between per-share prices and total debt.
* Correcting the input to use **Market Capitalization** resolved the error, reducing default probabilities to realistic levels for investment-grade firms.
* The improved model demonstrates that market participants view these firms as highly safe, with asset values significantly exceeding their debt obligations.

