# Market Risk Management | Multivariate GARCH & Backtesting

> A comprehensive market risk management framework for a four-asset ETF portfolio (SPY, TLT, GLD, USO) covering volatility modeling, VaR/ES estimation, rigorous backtesting, and Basel III regulatory compliance assessment.

---

## Dataset

- **Source:** Yahoo Finance (via `quantmod` / `yfinance`)
- **Assets:** SPY (S&P 500), TLT (20yr Treasury), GLD (Gold), USO (Crude Oil)
- **Period:** January 2021 – December 2025
- **Observations:** 1,253 daily closing prices
- **Train / Test Split:** First 1,003 obs (in-sample) | Last 250 obs (out-of-sample)
- **Risk-free Rate:** US 3-month T-bill (FRED: DTB3)

---

## Objectives

1. Model conditional volatility and dynamic correlations across asset classes
2. Estimate and backtest Value-at-Risk (VaR) and Expected Shortfall (ES)
3. Assess regulatory compliance under the Basel III framework
4. Construct and evaluate a rolling Tangency Portfolio using GARCH covariance forecasts

---

## Key Mathematical Concepts

### 1. Value-at-Risk (VaR)

VaR measures the **maximum loss** not exceeded at a given confidence level $1 - \alpha$ over a holding period.

$$VaR_{\alpha} = -\inf \{ x \in \mathbb{R} : P(L > x) \leq \alpha \}$$

Or equivalently, VaR is the $\alpha$-quantile of the loss distribution:

$$P(L > VaR_{\alpha}) = \alpha$$

**Interpretation:** At $\alpha = 1\%$, a daily VaR of $-2.79\%$ means there is a 1% probability the portfolio loses more than 2.79% in a single day.

---

### 2. Expected Shortfall (ES)

ES (also called **Conditional VaR** or **CVaR**) measures the **expected loss given that the loss exceeds VaR**. It is a coherent risk measure and is preferred over VaR under Basel III.

$$ES_{\alpha} = -\mathbb{E}[R \mid R \leq -VaR_{\alpha}] = -\frac{1}{\alpha} \int_0^{\alpha} VaR_u \, du$$

**Why ES over VaR?**
- VaR ignores the severity of losses beyond the threshold
- ES captures **tail risk** — the expected magnitude of extreme losses
- Basel III (FRTB) mandates ES at 97.5% confidence level as the primary risk measure

---

### 3. Portfolio VaR

For a portfolio with weight vector $\mathbf{w}$ and conditional covariance matrix $\mathbf{H}_t$:

$$\sigma^2_{p,t} = \mathbf{w}^\top \mathbf{H}_t \mathbf{w}$$

$$VaR_{\alpha,t} = z_\alpha \cdot \sigma_{p,t}$$

where $z_\alpha$ is the quantile of the assumed return distribution (Normal or Student-t).

---

### 4. Tangency Portfolio (Mean-Variance Optimization)

The Tangency Portfolio maximizes the **Sharpe Ratio**:

$$\max_{\mathbf{w}} \quad SR = \frac{\mathbf{w}^\top \boldsymbol{\mu} - r_f}{\sqrt{\mathbf{w}^\top \mathbf{H}_t \mathbf{w}}}$$

$$\text{subject to} \quad \mathbf{w}^\top \mathbf{1} = 1, \quad \mathbf{w} \geq 0$$

The analytical solution (unconstrained) is:

$$\mathbf{w}^* \propto \mathbf{H}_t^{-1} (\boldsymbol{\mu} - r_f \mathbf{1})$$

In this project, weights are rebalanced **daily** using one-step-ahead covariance forecasts from GARCH models, with strict **no-lookahead bias** enforcement.

---

## Performance Metrics

### Sharpe Ratio
Measures risk-adjusted return relative to a risk-free benchmark:

$$SR = \frac{R_p - r_f}{\sigma_p}$$

| Symbol | Description |
|--------|-------------|
| $R_p$ | Portfolio annualized return |
| $r_f$ | Risk-free rate (OOS T-bill: 1.15%) |
| $\sigma_p$ | Portfolio annualized volatility |

> **Result:** SR = **2.11** — well above the 1.0 threshold considered excellent.

---

### Sortino Ratio
Similar to Sharpe but penalizes only **downside volatility** $\sigma_d$:

$$\text{Sortino} = \frac{R_p - r_f}{\sigma_d}$$

> **Result:** Sortino = **2.64** > Sharpe = **2.11**, confirming upside volatility dominates downside — a desirable property.

---

### Maximum Drawdown (MDD)
Measures the largest peak-to-trough decline over the OOS period:

$$MDD = \min_{t} \frac{V_t - \max_{s \leq t} V_s}{\max_{s \leq t} V_s}$$

> **Result:** MDD = **−8.09%** — modest drawdown reflecting effective cross-asset diversification.

---

### Calmar Ratio
Return per unit of drawdown risk:

$$\text{Calmar} = \frac{R_p^{\text{ann}}}{|MDD|}$$

> **Result:** Calmar = **4.28** — high ratio indicates strong return relative to tail-loss risk.

---

### Portfolio Performance Summary (OOS 250-day)

| Metric | Value |
|--------|-------|
| Total Return | 34.35% |
| Annualised Return | 34.63% |
| Annualised Volatility | 15.87% |
| Sharpe Ratio | 2.1092 |
| Sortino Ratio | 2.6408 |
| Maximum Drawdown | −8.09% |
| Calmar Ratio | 4.2793 |
| Win Rate | 60.40% |
| Empirical VaR (1%) | −2.79% |
| Empirical VaR (5%) | −1.40% |

---

## VaR Backtesting Framework

### Statistical Tests

#### Unconditional Coverage Test (UC) — Kupiec (1995)
Tests whether the **observed violation rate** equals the expected rate $\alpha$:

$$H_0: \hat{\pi} = \alpha \quad \text{where} \quad \hat{\pi} = \frac{\text{Number of violations}}{T}$$

$$LR_{UC} = -2\ln\left[\frac{\alpha^{V}(1-\alpha)^{T-V}}{\hat{\pi}^{V}(1-\hat{\pi})^{T-V}}\right] \sim \chi^2(1)$$

---

#### Conditional Coverage Test (CC) — Christoffersen (1998)
Tests both correct coverage **and independence** of violations (no clustering):

$$LR_{CC} = LR_{UC} + LR_{ind} \sim \chi^2(2)$$

---

#### Dynamic Quantile Test (DQ) — Engle & Manganelli (2004)
Regresses the **hit sequence** $H_t = \mathbf{1}[R_t < -VaR_t] - \alpha$ on its own lags and $VaR_t$:

$$H_t = \beta_0 + \sum_{i=1}^{p} \beta_i H_{t-i} + \beta_{p+1} VaR_t + \varepsilon_t$$

$H_0$: all $\beta = 0$ (violations are unpredictable). Failure indicates **violation clustering**.

---

#### Fissler-Ziegel (FZ) Joint Loss Score
A **proper scoring rule** for simultaneous VaR and ES evaluation:

$$S_{FZ}(\hat{q}, \hat{e}; r) = \frac{1}{\hat{e}}\left(\hat{q} - r\right)\mathbf{1}[r \leq \hat{q}] - \frac{r}{\hat{e}} + \ln(-\hat{e}) + 1$$

Lower FZ score = better joint VaR/ES accuracy.

> **Champion:** DCC-GARCH Student-t — FZ Mean = **0.002955** (lowest overall)

---

#### Diebold-Mariano Test
Tests whether two models have **statistically equal predictive accuracy**:

$$DM = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})}} \sim \mathcal{N}(0,1)$$

where $d_t = S_1(t) - S_2(t)$ is the loss differential between two models. HAC-robust variance $\hat{V}$ is used to account for serial correlation.

> All pairwise DM tests confirm DCC-GARCH Student-t is statistically superior (DM ≈ −21 to −24, p = 0.000).

---

## 🏦 Basel III Regulatory Framework

### Traffic Light Backtesting

Under **Basel III / FRTB**, banks must backtest their internal VaR models daily at the **99% confidence level** (α = 1%) over a **250-day rolling window**. The number of exceptions (days where actual loss exceeds VaR) determines the **capital multiplier k**:

| Zone | Exceptions | Capital Multiplier k | Implication |
|------|-----------|----------------------|-------------|
| Green | 0 – 4 | 3.00 | Minimum capital requirement |
| Yellow | 5 – 9 | 3.40 – 3.85 | Increased capital charge |
| Red | ≥ 10 | 4.00 | Mandatory model review |

### Regulatory Capital Formula

Market Risk Capital Requirement under the Internal Models Approach (IMA):

$$MRC_t = \max\left(VaR_{t-1},\ k \cdot \frac{1}{60}\sum_{i=1}^{60} VaR_{t-i}\right) + SRC_t$$

where:
- $VaR_{t-1}$ = previous day's VaR at 99% confidence
- $k$ = multiplier based on backtesting zone (3.00–4.00)
- $SRC_t$ = Stressed Risk Charge (based on stressed VaR period)

### Why ES Replaced VaR in Basel III (FRTB)?

Under the **Fundamental Review of the Trading Book (FRTB, 2019)**:
- VaR at 99% is replaced by **ES at 97.5%** as the primary internal risk measure
- ES is preferred because it is a **coherent risk measure** (satisfies subadditivity) and captures the full shape of the tail beyond the threshold
- Banks must also demonstrate **P&L attribution** and pass both VaR and ES backtests
> **Key finding:** Student-t specifications consistently achieve Green zone status. Normal distribution models underestimate tail risk, leading to Yellow zone classification and 17% higher capital requirements.

---

## Model Recommendation

| Dimension | Champion Model |
|-----------|---------------|
| Overall VaR/ES Accuracy | DCC-GARCH Student-t (FZ = 0.002955) |
| Volatility Forecasting | GO-GARCH (MAE_Sigma 25–45% lower) |
| Correlation Forecasting | DCC-GARCH (5/6 pairs, lower MAE/MSE) |
| Basel III Compliance | DCC-GARCH Student-t — Green zone, k = 3.00 |
| Challenger Model | Parametric Student-t (simpler, 100% UC/CC pass) |

---

## Tools & Libraries

| Category | Tools |
|----------|-------|
| Language | R |
| GARCH Modeling | `rugarch`, `rmgarch` |
| Data | `quantmod`, `xts`, `zoo` |
| Backtesting | `GAS`, custom rolling backtest functions |
| Portfolio Optimization | Custom Tangency Portfolio via `quadprog` |
| Visualization | `ggplot2`, `gridExtra` |

---

---

## 📚 References

- Engle, R. F. (2002). Dynamic Conditional Correlation. *Journal of Business & Economic Statistics.*
- Van der Weide, R. (2002). GO-GARCH: A Multivariate Generalized Orthogonal GARCH Model. *Journal of Applied Econometrics.*
- Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. *Journal of Derivatives.*
- Christoffersen, P. (1998). Evaluating Interval Forecasts. *International Economic Review.*
- Fissler, T. & Ziegel, J. F. (2016). Higher Order Elicitability and Osband's Principle. *Annals of Statistics.*
- Basel Committee on Banking Supervision (2019). *Minimum Capital Requirements for Market Risk (FRTB).*
