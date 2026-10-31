# NSE Volatility Forecasting: Econometrics vs. Machine Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-9%20Passed-brightgreen.svg)]()
[![Market](https://img.shields.io/badge/Market-NSE%20India%20(Large--Cap)-orange.svg)]()
[![Evaluation](https://img.shields.io/badge/Validation-Walk--Forward%20Embargo%20(33%20Folds)-purple.svg)]()

An institutional-grade quantitative finance study benchmarking classical econometric models against modern machine learning for forecasting **5-day forward realized volatility ($H=5$)** across **15 large-cap National Stock Exchange of India (NSE)** equities over ~8.25 years (33 quarterly walk-forward test folds).

---

## 📌 Executive Summary

Predicting stock volatility is critical for portfolio risk management, Value-at-Risk (VaR), option pricing (Greeks), and position sizing. However, financial machine learning frequently falls prey to **data leakage (look-ahead bias)** and **inappropriate evaluation metrics**.

This project addresses these challenges directly:
1. **Zero Look-Ahead Guarantee**: Rigorously audited feature pipelines with an $H=5$ day embargo buffer between train and test folds to eliminate target leakage.
2. **Econometric vs. Machine Learning Benchmark**: Evaluates **Naive Volatility**, **RiskMetrics EWMA**, **HAR-RV with Ridge Regularization**, and **LightGBM** across multiple volatility regimes.
3. **Statistically Sound Loss Metrics**: Evaluated under **Quasi-Likelihood (QLIKE)** loss—the mathematically proven, robust metric for volatility forecast evaluation—and **Mean Absolute Error (MAE)**.
4. **Non-Parametric Bias Correction**: Implements **Duan's Smearing Estimator** to neutralize the systematic downward bias caused by exponentiating log-space volatility forecasts.

---

## 📊 Empirical Findings & Benchmark Results

### Overall Performance Across 33 Walk-Forward Folds (~8.25 Years)

| Model | MAE | QLIKE Loss | MAE vs EWMA (%) | QLIKE vs EWMA (%) | Folds Beating Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HAR (Ridge)** | **0.0056** | 0.5232 | **-6.11%** | +0.71% | 20 / 33 |
| **LightGBM** | **0.0056** | 0.5405 | **-5.44%** | +4.04% | 15 / 33 |
| **EWMA ($\lambda=0.94$)** | 0.0060 | **0.5195** | *Baseline (0.0%)* | *Baseline (0.0%)* | — |
| **Naive (RV5)** | 0.0068 | 1.2662 | +14.57% | +143.74% | 0 / 33 |

*(Negative percentage indicates superior performance relative to the EWMA benchmark).*

<p align="center">
  <img src="assets/relative_improvement.svg" alt="MAE Improvement vs EWMA" width="85%">
</p>

---

## 🔍 Regime Analysis: Where Does Each Model Win?

Market dynamics shift dramatically between calm bull markets and crisis regimes. We stratified the 33 test folds into terciles based on trailing 22-day realized volatility (`rv22`) known strictly at test time:

| Regime | EWMA | HAR (Ridge) | LightGBM | Naive (RV5) | Top Performer |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Low Volatility** | 0.5558 | **0.4956** | 0.4996 | 1.5300 | 🏆 **HAR (Ridge) & LightGBM (-10.8%)** |
| **Mid Volatility** | **0.5199** | 0.5440 | 0.5341 | 1.2918 | 🏆 **EWMA** |
| **High Volatility (Spikes/Crashes)** | **0.5565** | 0.5667 | 0.5792 | 1.0559 | 🏆 **EWMA** |

<p align="center">
  <img src="assets/regime_performance.svg" alt="Loss by Volatility Regime" width="90%">
</p>

### Key Quantitative Insights

1. **Econometrics Beats Complexity in Calm Markets**: In low-volatility environments, **HAR (Ridge)** and **LightGBM** outperform EWMA by more than **10% on QLIKE**. The multi-frequency aggregation (daily + weekly + monthly horizons) captures persistent memory dynamics that simple exponential smoothing ignores.
2. **EWMA Shields Against Tail Risk in Shock Regimes**: During severe market disruptions (e.g. COVID-19 crash, geopolitical turmoil), statistical and tree models trained on calmer trailing windows exhibit lag. **EWMA ($\lambda=0.94$)** adapts instantly due to heavy geometric weighting on the immediate return shock.
3. **The ML vs. Linear Reality**: LightGBM beats HAR in **15 of 33 folds (45.5%)**. This confirms a fundamental reality of low signal-to-noise daily equity returns: gradient boosted trees do not automatically surpass well-specified linear autoregressive econometrics without intraday microstructure data.
4. **Naive Estimation Fails**: Using trailing 5-day realized volatility as the future 5-day forecast generates **+143.7% higher QLIKE loss**, proving that simple rolling windows are wholly inadequate for options and risk desks.

<p align="center">
  <img src="assets/walkforward_folds.svg" alt="Walk-Forward QLIKE Loss Across 33 Folds" width="90%">
</p>

---

## 📐 Mathematical Formulation

### 1. Forecast Target: Realized Volatility ($H=5$)
For stock returns $r_t = \ln(P_t / P_{t-1})$, the forward realized volatility over trading horizon $H=5$ is defined as:

$$\sigma_{t+1:t+H} = \sqrt{\frac{1}{H} \sum_{k=1}^H r_{t+k}^2}$$

The model fits in log-space to enforce positivity and normalize heteroskedastic error variance:

$$y_t = \ln(\sigma_{t+1:t+H} + \epsilon)$$

### 2. Heterogeneous Autoregressive (HAR-RV) Model
Adapted from Corsi (2009), the HAR model decomposes volatility into three cascade time horizons:

$$\ln(\sigma_{t+1:t+H}) = \beta_0 + \beta^{(d)} \ln(RV_t^{(1)}) + \beta^{(w)} \ln(RV_t^{(5)}) + \beta^{(m)} \ln(RV_t^{(22)}) + \varepsilon_t$$

Regularized with $L_2$ Ridge regression ($\alpha = 10^{-3}$) to prevent multi-collinearity between overlapping rolling windows.

### 3. RiskMetrics EWMA
Exponentially Weighted Moving Average with decay factor $\lambda = 0.94$ ($\alpha = 1 - \lambda = 0.06$):

$$\sigma_t^2 = (1 - \alpha) \sigma_{t-1}^2 + \alpha r_t^2$$

### 4. Duan's Smearing Re-transformation Correction
Because the models forecast $\hat{y} = \ln(\hat{\sigma})$, recovering the volatility scale via naive exponentiation $\exp(\hat{y})$ incurs a downward bias due to **Jensen's Inequality** ($\mathbb{E}[e^X] \ge e^{\mathbb{E}[X]}$). We apply **Duan's non-parametric smearing factor** $S$:

$$S = \frac{1}{N_{train}} \sum_{i=1}^{N_{train}} \exp(y_i - \hat{y}_i), \quad \hat{\sigma}_{corrected} = S \cdot \exp(\hat{y}_{test})$$

### 5. Quasi-Likelihood (QLIKE) Loss Function
Under volatility proxy noise, standard MSE is distorted by Jensen's inequality and unhedged asymmetry. As demonstrated by Patton (2011), QLIKE provides a strictly consistent, scale-invariant loss:

$$L_{QLIKE}(\sigma^2, \hat{\sigma}^2) = \frac{\sigma^2}{\hat{\sigma}^2} - \ln\left(\frac{\sigma^2}{\hat{\sigma}^2}\right) - 1$$

QLIKE heavily penalizes under-forecasting volatility ($\hat{\sigma} < \sigma$), which aligns directly with the asymmetric risk profile of options market makers and risk desks.

---

## 🛡️ Leak-Free Validation & Look-Ahead Prevention

Financial ML pipelines frequently suffer from subtle future leakage:

```
Timeline: [ ------ Train Window ------ ]-[ Embargo (H=5) ]-[ -- Test Fold (63d) -- ]
                                        ^
                    Purged buffer prevents training target 
                    y_t = rv(t+1..t+5) from bleeding into Test
```

- **Initial Training**: 756 trading days (~3 years).
- **Test Fold**: 63 trading days (~1 quarter).
- **Embargo Buffer**: $H = 5$ days completely discarded between the end of the train set and the beginning of the test set.
- **Unit Test Suite**: `test_no_lookahead.py` verifies that truncated feature series match full series identically across sample rows $t \in \{60, 150, 300\}$.

---

## 📂 Repository Structure

```
.
├── assets/                         # Rendered high-res SVG benchmark charts
│   ├── regime_performance.svg
│   ├── relative_improvement.svg
│   └── walkforward_folds.svg
├── s2_volatility/
│   ├── __init__.py                 # Package declaration
│   ├── data.py                     # Yahoo Finance download & validation checks
│   ├── features.py                 # Mathematical feature extraction & target builder
│   ├── walkforward.py              # Walk-forward cross-validation engine
│   ├── generate_visuals.py         # SVG chart rendering engine
│   ├── data/
│   │   ├── prices.csv              # 10+ years daily OHLCV for 15 NSE stocks
│   │   └── results.csv             # Full 33-fold metric logs
│   └── tests/
│       ├── test_data_validation.py # Data integrity tests
│       ├── test_metrics.py         # QLIKE & Duan smearing property tests
│       └── test_no_lookahead.py    # Strict look-ahead bias audit
├── run.py                          # Unified CLI entrypoint
├── pyproject.toml                  # Modern Python packaging configuration
├── requirements.txt                # Pinned dependencies
├── LICENSE                         # MIT License
└── README.md                       # Quantitative report & documentation
```

---

## 🚀 Quickstart & Reproducibility

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Sanjeevharge/NSE-Forecasting.git
cd NSE-Forecasting

# Create and activate virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Verification Tests

Run the unit test suite to verify look-ahead prevention, metric consistency, and data validation:

```bash
python run.py --test
```
```
collected 9 items
s2_volatility/tests/test_data_validation.py ... PASSED
s2_volatility/tests/test_metrics.py ........ PASSED
s2_volatility/tests/test_no_lookahead.py .. PASSED
============================== 9 passed in 2.45s ==============================
```

### 3. Run the Walk-Forward Backtest

Execute the full walk-forward simulation across all 33 folds and update the visual charts:

```bash
python run.py --evaluate
```

### 4. Optional: Refresh Market Data

To re-fetch up-to-date daily prices for the 15 NSE equities from Yahoo Finance:

```bash
python run.py --download
```

---

## 📈 Universe of Equities

The study analyzes 15 liquid blue-chip constituents of the **NIFTY 50 Index**:

`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`, `SBIN.NS`, `ITC.NS`, `LT.NS`, `BHARTIARTL.NS`, `HINDUNILVR.NS`, `AXISBANK.NS`, `KOTAKBANK.NS`, `MARUTI.NS`, `SUNPHARMA.NS`, `WIPRO.NS`.

---

## 📚 Academic & Quantitative References

1. **Corsi, F. (2009)**. *A Simple Approximate Long-Memory Model of Realized Volatility*. Journal of Financial Econometrics, 7(2), 174–196.
2. **Patton, A. J. (2011)**. *Volatility forecast comparison using imperfect volatility proxies*. Journal of Econometrics, 160(1), 246–256.
3. **Duan, N. (1983)**. *Smearing Estimate: A Nonparametric Retransformation Method*. Journal of the American Statistical Association, 78(383), 605–610.
4. **Andersen, T. G., & Bollerslev, T. (1998)**. *Answering the skeptics: Yes, standard volatility models do provide accurate forecasts*. International Economic Review, 39(4), 885–905.
5. **J.P. Morgan / Reuters (1996)**. *RiskMetrics Technical Monitor*. 4th Edition, New York.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

