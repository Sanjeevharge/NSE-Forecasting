# NSE Volatility Forecasting: Econometrics vs. Machine Learning

A benchmark comparing classical econometric models against gradient boosted decision trees for 5-day forward realized volatility ($H=5$) across 15 large-cap equities on the National Stock Exchange of India (NSE).

Evaluated across 33 quarterly walk-forward test folds (~8.25 years of daily data) with an embargo buffer to prevent look-ahead bias.

---

## Overview

Volatility forecasting is standard in risk management (Value-at-Risk), derivative pricing, and portfolio optimization. This study evaluates whether non-linear machine learning (LightGBM) outperforms standard quantitative baselines (RiskMetrics EWMA, HAR-RV) under realistic, leak-free conditions and asymmetric evaluation metrics.

### Models Evaluated

1. **Naive Baseline (RV5)**: Trailing 5-day realized volatility.
2. **EWMA ($\lambda=0.94$)**: RiskMetrics exponential moving average benchmark.
3. **HAR (Ridge)**: Heterogeneous Autoregressive model with daily, weekly, and monthly volatility components and $L_2$ shrinkage.
4. **LightGBM**: Gradient boosted regression trees using multi-scale volatility, signed return (leverage effect), and normalized volume ratios.

---

## Benchmark Results

### Overall Performance (33 Walk-Forward Folds, ~8.25 Years)

| Model | MAE | QLIKE Loss | MAE vs EWMA (%) | QLIKE vs EWMA (%) | Folds Beating Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HAR (Ridge)** | **0.0056** | 0.5232 | **-6.11%** | +0.71% | 20 / 33 |
| **LightGBM** | **0.0056** | 0.5405 | **-5.44%** | +4.04% | 15 / 33 |
| **EWMA** | 0.0060 | **0.5195** | Baseline | Baseline | — |
| **Naive (RV5)** | 0.0068 | 1.2662 | +14.57% | +143.74% | 0 / 33 |

*Note: For both MAE and QLIKE, lower values indicate better performance.*

### Performance by Volatility Regime (Mean QLIKE)

Folds are grouped into terciles based on 22-day realized volatility at test time:

| Regime | EWMA | HAR (Ridge) | LightGBM | Naive (RV5) | Lowest Error |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Low Volatility** | 0.5558 | **0.4956** | 0.4996 | 1.5300 | HAR (Ridge) / LightGBM |
| **Mid Volatility** | **0.5199** | 0.5440 | 0.5341 | 1.2918 | EWMA |
| **High Volatility** | **0.5565** | 0.5667 | 0.5792 | 1.0559 | EWMA |

### Key Observations

- **Calm regimes favor multi-scale models**: In low-volatility conditions, HAR and LightGBM reduce QLIKE loss by ~10% compared to EWMA, as multi-horizon signals capture persistent volatility clustering.
- **Shock regimes favor EWMA**: In high-volatility periods, EWMA achieves the lowest QLIKE loss because its exponential weighting adapts rapidly to immediate price shocks without relying on pre-crisis training windows.
- **ML vs. linear econometrics**: LightGBM outperformed HAR on QLIKE in 15 of 33 folds (45.5%). In daily financial return data with low signal-to-noise ratios, well-specified linear models remain highly competitive with decision trees.
- **Trailing window limits**: Trailing 5-day volatility performs poorly as a direct forward forecast (+143% higher QLIKE loss).

---

## Methodology

### Target Formulation
For daily log returns $r_t = \ln(P_t / P_{t-1})$, the 5-day forward realized volatility is:

$$\sigma_{t+1:t+H} = \sqrt{\frac{1}{H} \sum_{k=1}^H r_{t+k}^2}, \quad H=5$$

Models are fit on the log transform $y_t = \ln(\sigma_{t+1:t+H} + \epsilon)$.

### Duan Smearing Correction
Because models forecast in log-space, simple exponentiation $\exp(\hat{y})$ underestimates expected volatility due to Jensen's inequality. Duan's non-parametric smearing estimator corrects this:

$$S = \frac{1}{N} \sum_{i=1}^N \exp(y_i - \hat{y}_i), \quad \hat{\sigma} = S \cdot \exp(\hat{y})$$

### Loss Functions
- **MAE**: Mean Absolute Error on the volatility scale ($|\sigma - \hat{\sigma}|$).
- **QLIKE**: Quasi-Likelihood loss, an asymmetric metric robust to volatility proxy noise (Patton, 2011):

$$L_{QLIKE}(\sigma^2, \hat{\sigma}^2) = \frac{\sigma^2}{\hat{\sigma}^2} - \ln\left(\frac{\sigma^2}{\hat{\sigma}^2}\right) - 1$$

### Walk-Forward Validation and Embargo
- Initial training window: 756 trading days (~3 years).
- Test fold size: 63 trading days (~1 quarter).
- **Embargo**: An $H=5$ day gap between the training cutoff and test start ensures target windows from training do not overlap with test observations.
- Unit tests verify that features at day $t$ remain identical whether computed on the full history or truncated at $t$.

---

## Project Structure

```
.
├── assets/                         # SVG performance figures
├── s2_volatility/
│   ├── data.py                     # Price download and data validation
│   ├── features.py                 # Feature definitions and target calculation
│   ├── walkforward.py              # Walk-forward cross-validation engine
│   ├── generate_visuals.py         # SVG chart generator
│   ├── data/
│   │   ├── prices.csv              # Historical daily OHLCV data
│   │   └── results.csv             # 33-fold evaluation logs
│   └── tests/
│       ├── test_no_lookahead.py    # Look-ahead bias verification
│       ├── test_metrics.py         # QLIKE and Duan smearing tests
│       └── test_data_validation.py # Data integrity tests
├── run.py                          # CLI runner for test, evaluate, download
├── requirements.txt                # Dependencies
├── pyproject.toml                  # Package configuration
└── LICENSE                         # MIT License
```

---

## Getting Started

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/Sanjeevharge/NSE-Forecasting.git
cd NSE-Forecasting

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tests

Verify data validation, metric behavior, and look-ahead prevention:

```bash
python run.py --test
```

### 3. Run Walk-Forward Evaluation

Run the 33-fold backtest and display the performance tables:

```bash
python run.py --evaluate
```

### 4. Optional: Update Market Data

Download fresh historical data for the 15 NSE tickers:

```bash
python run.py --download
```

---

## Data Universe

15 large-cap constituents of the NIFTY index:
`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`, `SBIN.NS`, `ITC.NS`, `LT.NS`, `BHARTIARTL.NS`, `HINDUNILVR.NS`, `AXISBANK.NS`, `KOTAKBANK.NS`, `MARUTI.NS`, `SUNPHARMA.NS`, `WIPRO.NS`.

---

## References

- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility*. Journal of Financial Econometrics.
- Patton, A. J. (2011). *Volatility forecast comparison using imperfect volatility proxies*. Journal of Econometrics.
- Duan, N. (1983). *Smearing Estimate: A Nonparametric Retransformation Method*. JASA.
- J.P. Morgan / Reuters (1996). *RiskMetrics Technical Monitor*.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
