# BTC Forecast Terminal (AlphaI × Polaris)

A professional-grade probabilistic forecasting system designed to predict the 95% confidence range for Bitcoin (BTCUSDT) one hour into the future. Unlike standard point-prediction models, this terminal focuses on **calibrated uncertainty estimation** to support high-stakes decision making.

## 📌 Project Overview
The goal of this system is to provide actionable probabilistic insights under market uncertainty. It targets a **95% coverage rate** (actual price falling within the predicted range 95% of the time) while minimizing the interval width for maximum precision.

## ⚙️ Methodology
The system utilizes a refined Quantitative approach:
- **Model**: Geometric Brownian Motion (GBM) enhanced with **Student-t distribution (df=5)** to accurately capture the "fat tails" and extreme kurtosis characteristic of crypto markets.
- **Volatility Clustering**: Implements a rolling standard deviation (sigma) based on the most recent 30-hour log-returns, allowing the model to adapt to changing market regimes.
- **Simulation**: A Monte Carlo engine generates **10,000 potential price paths** for every forecast.
- **Calibration**: Applied a post-simulation **0.1% tightening adjustment** based on extensive backtesting to optimize the balance between coverage and forecast efficiency.

## 📊 Backtest Performance (30D)
Rigorous rolling-window backtests (no data leakage) yield the following performance metrics:
- **Backtest Coverage**: **0.9564** (Targeting 95%)
- **Average Range Width**: **1,353.37 USDT**
- **Forecast Efficiency (Winkler Score)**: **1815.98** (Minimized penalty for width and misses)

## 🚀 Terminal Features
- **Live Forecasting**: Real-time price ingestion and prediction range generation.
- **Decision Layer**: Automated **Suggested Action** (Strong Hold / Hold / No Trade) and **Confidence Indicator** based on relative interval width.
- **Visual Intelligence**: TradingView-style interactive Plotly chart with current/previous forecast bands and market phase delimiters.
- **Transparency**: A dedicated "Last Prediction Result" banner that evaluates the system's previous forecast against actual spot movement.

## ▶️ Local Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Launch the terminal
streamlit run app.py
```

## 🏁 Key Philosophy
> "In trading, probabilistic accuracy is more valuable than point prediction."

This system is built on the principle that understanding **what you don't know** is as important as knowing where the price is going.
