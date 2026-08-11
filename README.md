# AI/ML Real-Time Stock Screening & Trade Filter

An AI/ML-based stock screening and paper-trading system that combines technical indicators, market-depth signals, and machine learning to filter potential BUY/SELL crossover signals.

> **Note:** This project uses synthetic market data for testing and paper trading. It does not perform real-money trading.

## 🚀 Project Overview

The system is designed to identify potential trading opportunities using:

* SMMA 20 / SMMA 120 crossover
* LTQ (Last Traded Quantity)
* Bid/Ask market depth
* Bid/Ask imbalance
* Price and volume movement
* Random Forest machine learning
* AI probability-based trade filtering
* ACCEPT / AVOID decisions
* Trade deterioration monitoring
* Historical paper trading
* Strategy performance comparison
* Streamlit dashboard

The main objective is to **filter weak crossover signals rather than simply generate more trades**.

## 🏗️ System Architecture

```text
Market Data
     |
     v
Feature Engineering
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
         ML Feature Set
                |
                v
       Random Forest Model
                |
                v
        AI Probability
                |
                v
        ACCEPT / AVOID
                |
                v
         Paper Trading
                |
                v
      Strategy Comparison
                |
                v
      Streamlit Dashboard
```

## 🧠 Machine Learning

The project uses a **RandomForestClassifier** to classify whether a crossover signal should be accepted.

### ML Features

1. `smma_spread`
2. `smma_spread_change`
3. `ltq`
4. `ltq_change`
5. `ltq_acceleration`
6. `bid_quantity`
7. `ask_quantity`
8. `bid_ask_imbalance`
9. `bid_ask_spread`
10. `price_change`
11. `price_change_pct`
12. `volume`
13. `volume_change`

The historical target is generated using future price movement during training only. Future information is **not used as a live prediction feature**.

## 📊 Strategy Results

The synthetic paper-trading experiment produced the following comparison:

| Metric              | Basic SMMA | AI/ML Filter |
| ------------------- | ---------: | -----------: |
| Total Signals       |         34 |           34 |
| Executed Trades     |         34 |           20 |
| Avoided Signals     |          0 |           14 |
| Profitable Trades   |         28 |           20 |
| Losing Trades       |          6 |            0 |
| Win Rate            |     82.35% |         100% |
| Total Simulated P/L |    27.4383 |      22.1572 |

The AI/ML filter avoided all 6 losing trades in this particular synthetic test.

**Important:** These results are from a small synthetic dataset and should not be interpreted as live-market performance.

## 📈 Dashboard

The project includes a Streamlit dashboard displaying:

* LTP
* Current trading signal
* AI probability
* ACCEPT / AVOID decision
* LTQ
* Bid/Ask quantities
* Bid/Ask imbalance
* SMMA 20 / SMMA 120
* Recent crossover signals
* Paper-trading statistics

## 📁 Project Structure

```text
AI_ML_Stock_Trader/
│
├── comparison.py
├── dashboard.py
├── decision_test.py
├── generate_data.py
├── paper_test.py
├── train.py
├── validate.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── sample_data.csv
│   ├── processed_data.csv
│   ├── paper_trading_results.csv
│   ├── basic_smma_results.csv
│   └── ai_ml_results.csv
│
├── models/
│   └── trade_filter_model.pkl
│
└── src/
    ├── data_feed.py
    ├── features.py
    ├── indicators.py
    ├── ml_model.py
    ├── paper_trading.py
    └── signal_detector.py
```

## ⚙️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Plotly
* Joblib
* Git & GitHub

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/anushamangali61-stack/AI_ML_Stock_Trader.git
cd AI_ML_Stock_Trader
```

Create and activate a virtual environment:

### Windows

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## ▶️ Run the Project

Generate sample market data:

```powershell
python generate_data.py
```

Train the machine-learning model:

```powershell
python train.py
```

Run the paper-trading test:

```powershell
python paper_test.py
```

Compare the strategies:

```powershell
python comparison.py
```

Run the decision test:

```powershell
python decision_test.py
```

Validate the project:

```powershell
python validate.py
```

Launch the Streamlit dashboard:

```powershell
streamlit run dashboard.py
```

## 🔍 Example AI Decision

```text
AI Probability : 64.61%
Decision       : AVOID

Reason:
Strong bid-side support;
Price moving downward
```

The system combines the ML probability with market conditions before producing the final decision.

## 🧪 Validation

The project validation checks:

* Required files
* Dataset availability
* Required market-data columns
* ML model loading
* Paper-trading results
* Basic strategy results
* AI/ML strategy results

Current validation:

```text
Dataset loaded successfully
Rows: 5000
Columns: 15

ML model:
RandomForestClassifier

Paper trades:
20

Basic strategy trades:
34

AI/ML trades:
20
```

## 🎯 Future Improvements

* Connect to a real-time market-data API
* Add multiple stocks
* Improve class balancing
* Increase historical training data
* Add walk-forward validation
* Add transaction costs and slippage
* Add risk management
* Add stop-loss and take-profit logic
* Add real-time dashboard updates
* Deploy the dashboard to the cloud

## ⚠️ Disclaimer

This project is intended for **educational and research purposes**.

It uses synthetic market data and paper trading. It does not provide financial advice and does not execute real-money trades.
