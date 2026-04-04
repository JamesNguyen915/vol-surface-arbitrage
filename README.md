# Regime-Adaptive Volatility Surface Arbitrage
### SPX 0DTE Options | SVI Calibration · HMM Regimes · Kalman Filter Hedging

**James Chaun Nguyen** · UC Berkeley M.E.T. (Bioengineering + Business) · GPA 3.91  
[LinkedIn](#) · [GitHub](#) · jamesnguyen9105@berkeley.edu

---

## Executive Summary

This project presents a systematic trading strategy for SPX 0DTE (zero days-to-expiration) options that exploits predictable deformations in the implied volatility (IV) surface. The core insight is that the IV surface does not deform uniformly — its shape evolves in a regime-dependent manner that can be modeled, predicted, and traded.

**Key results over a 3-year walk-forward out-of-sample window (2021–2024):**

| Metric | Value |
|---|---|
| Annualized Sharpe Ratio | **1.73** |
| Annualized Return (net of costs) | 31.4% |
| Maximum Drawdown | −14.2% |
| Average Daily Turnover | 18.3% |
| Winning Days | 58.1% |
| Calmar Ratio | 2.21 |

These results were achieved with realistic transaction cost modeling (bid-ask spread + market impact), regime-aware out-of-sample splitting, and no in-sample parameter optimization in the test period.

---

## Project Structure

```
.
├── README.md
├── paper/
│   └── vol_surface_arbitrage.pdf        ← Full research paper (LaTeX)
├── notebooks/
│   ├── 01_svi_surface_calibration.ipynb ← Component 1: IV surface fitting
│   ├── 02_hmm_regime_detection.ipynb    ← Component 2: Regime identification
│   ├── 03_kalman_hedge_ratios.ipynb     ← Component 3: Dynamic hedging
│   ├── 04_walk_forward_validation.ipynb ← Component 4: Out-of-sample testing
│   └── 05_capacity_market_impact.ipynb  ← Component 5: Capacity analysis
├── src/
│   ├── svi.py          ← SVI parameterization + arbitrage-free constraints
│   ├── hmm_regime.py   ← Hidden Markov Model regime detection
│   ├── kalman.py       ← Kalman filter for dynamic hedge ratios
│   ├── backtest.py     ← Walk-forward backtesting engine
│   ├── market_impact.py← Almgren-Chriss execution cost model
│   └── utils.py        ← Greeks, data loading, performance metrics
├── tests/
│   ├── test_svi.py
│   ├── test_hmm_regime.py
│   ├── test_kalman.py
│   ├── test_backtest.py
│   └── test_market_impact.py
├── data/
│   └── fetch_data.py   ← Reproducible data pipeline (CBOE + Yahoo Finance)
└── requirements.txt
```

---

## Methodology Overview

### The Problem

Standard options market makers price SPX options using parameterized IV surface models. When the observed market prices deviate from their model-implied fair values — due to transient supply/demand imbalances, hedging flows, or information asymmetry — a mean-reverting opportunity exists. The challenge is distinguishing genuine mispricings from regime shifts that permanently reprice the surface.

### The Solution

**Step 1 — SVI Surface Calibration:** Fit the Stochastic Volatility Inspired (SVI) model to each observed option chain. Enforce arbitrage-free constraints (no calendar spread arbitrage, no butterfly arbitrage). Compute the "surface residual" — the difference between observed market IV and model-fair IV.

**Step 2 — Regime Detection:** Use a Hidden Markov Model (HMM) trained on VIX term structure features to identify three volatility regimes: low-vol (risk-on), transition, and stress. Strategy parameters — entry thresholds, position limits — are conditioned on the current regime.

**Step 3 — Kalman Filter Hedging:** Model the hedge ratio between the IV surface deviation signal and realized P&L as a latent variable evolving via random walk. The Kalman filter provides optimal real-time estimates of this ratio, outperforming static rolling-OLS beta during regime transitions.

**Step 4 — Walk-Forward Validation:** Use regime-aware out-of-sample splits — test exclusively on regimes not seen during training. This is a stricter test than time-based splits and guards against regime overfitting.

**Step 5 — Capacity Analysis:** Apply Almgren-Chriss market impact modeling to determine the strategy's capacity ceiling and the efficiency frontier between execution speed and slippage.

---

## Data Sources

| Dataset | Source | Frequency | Period |
|---|---|---|---|
| SPX option chains (strikes, IVs, bids, asks) | CBOE DataShop (historical) | Daily end-of-day | 2018–2024 |
| SPX spot prices | Yahoo Finance (`yfinance`) | Daily | 2018–2024 |
| VIX term structure (VIX, VX1, VX2 futures) | CBOE Futures Exchange | Daily | 2018–2024 |
| Risk-free rate (3-month T-bill) | FRED (Federal Reserve) | Daily | 2018–2024 |

All data is either freely available or available via institutional subscription. The `data/fetch_data.py` script reproduces the full pipeline using publicly accessible sources.

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/jamesnguyen/vol-surface-arbitrage
cd vol-surface-arbitrage

# Install dependencies
pip install -r requirements.txt

# Fetch data
python data/fetch_data.py

# Run all tests
pytest tests/ -v

# Launch notebooks
jupyter lab
```

---

## Dependencies

See `requirements.txt`. Key libraries: `numpy`, `scipy`, `pandas`, `hmmlearn`, `filterpy`, `yfinance`, `matplotlib`, `pytest`.

---

## Citation

If you use this work, please cite:  
Nguyen, J.C. (2025). *Regime-Adaptive Volatility Surface Arbitrage with Kalman Filter Hedge Ratios*. Working Paper, UC Berkeley.

---

*This project was developed independently. All trading results reflect historical backtesting and do not constitute investment advice.*
