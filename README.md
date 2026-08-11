# AI/ML Real-Time Stock Screening & Trade Filter

A Python-based AI/ML stock screening and paper-trading prototype that combines technical indicators, market-depth features, and machine learning to filter potential trading signals.

> ⚠️ This project uses synthetic market data for testing and demonstration. It does not perform real-money trading.

---

## Project Overview

The system is designed to detect SMMA crossover signals and evaluate whether a signal should be accepted or avoided using additional market features and a machine-learning model.

The project combines:

- SMMA 20
- SMMA 120
- Crossover detection
- LTQ (Last Traded Quantity)
- Bid/Ask market depth
- Bid/Ask imbalance
- Price movement
- Volume movement
- Machine Learning probability
- ACCEPT / AVOID decision
- Trade deterioration monitoring
- Paper trading
- Strategy comparison
- Streamlit dashboard

---

## System Architecture

```text
Market Data
     |
     v
Data Feed
     |
     v
SMMA 20 / SMMA 120
     |
     v
Crossover Detection
     |
     +----------------------+
     |                      |
     v                      v
LTQ Analysis          Bid/Ask Analysis
     |                      |
     +----------+-----------+
                |
                v
        Feature Engineering
                |
                v
          Random Forest ML
                |
                v
       Probability Prediction
                |
                v
         ACCEPT / AVOID
                |
                v
       Paper Trading Engine
                |
                v
      Strategy Comparison
                |
                v
       Streamlit Dashboard