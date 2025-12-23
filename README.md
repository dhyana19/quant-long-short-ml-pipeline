# Enhanced Long–Short Trading Strategy via Quant ML Pipeline

An end-to-end machine learning–driven long–short trading framework for Indian equity markets (NSE), combining alpha feature engineering, LightGBM models with time-series validation, and an ML-based exit & risk recommendation engine.

This project focuses on **signal quality, risk-adjusted performance, and disciplined exits**, rather than naïve prediction accuracy.

---

## Project Overview

The pipeline fetches historical OHLCV data directly from **Angel One SmartAPI**, engineers a diverse set of technical and statistical features, and trains gradient-boosted models to generate **high-confidence long/short signals**.

A second ML system models **trade exits and risk**, predicting holding period, success probability, expected return, and stop-loss behavior, and converts them into actionable recommendations.

---

## Key Components

### 1. Long–Short Signal Generation
- Processes **10+ years of OHLCV data** for ~100 NSE stocks
- Engineers **50+ alpha features**, including:
  - RSI, ADX, ATR, Supertrend
  - Volatility & volume-based indicators
  - Momentum and trend strength features
- Trains **LightGBM classifiers**
- Uses **Optuna with time-series cross-validation (TSCV)** for robust hyperparameter tuning
- Outputs probabilistic **LONG / SHORT signals** with improved precision and win-rate

Notebook:  
`notebooks/01_signal_generation.ipynb`

---

### 2. Exit Strategy & Risk Assessment Engine
- ML-driven exit modeling using LightGBM to predict:
  - Optimal holding period
  - Trade success probability
  - Expected returns
  - Stop-loss trigger likelihood
- Combines model outputs with rule-based filters:
  - Profit targets
  - Trailing stop-loss
  - RSI & momentum reversals
- Produces interpretable trade recommendations:
  - **STRONG**
  - **MODERATE**
  - **RISKY**
  - **AVOID**

Notebook:  
`notebooks/02_exit_strategy_risk_engine.ipynb`

---

## Tech Stack

- **Languages:** Python  
- **ML / Optimization:** LightGBM, Optuna, Scikit-learn  
- **Feature Engineering:** Pandas, Pandas-TA, NumPy, SciPy  
- **Visualization:** Matplotlib  
- **Market Data:** Angel One SmartAPI (SmartConnect)

---

## Data Source

This project does **not ship any dataset**.

All OHLCV data is fetched **at runtime** using Angel One SmartAPI.  
No historical market data is stored or version-controlled in this repository.

---

## Repository Structure

quant-long-short-ml-pipeline/
├─ notebooks/
│ ├─ 01_signal_generation.ipynb
│ └─ 02_exit_strategy_risk_engine.ipynb
├─ src/ # (modularization planned)
├─ configs/
│ └─ config.example.yaml
├─ assets/
│ └─ screenshots/
├─ data/
│ └─ README.md
├─ README.md
├─ requirements.txt
├─ .gitignore
└─ LICENSE

---

## Disclaimer

This project is for **educational and research purposes only**.  
It does **not** constitute financial or investment advice.

---

## Author

**Dhyana Parmar**  
Undergraduate | Data Science & Quantitative ML
