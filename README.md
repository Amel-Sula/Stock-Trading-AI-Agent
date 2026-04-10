# Intelligent Trading Agent Using Machine Learning (CEN352 Term Project)

## Group Members & Roles
- **Dion Toska** — Data preparation & features
- **Aldinio Nurce** — Model training & evaluation
- **Amel Sula** — Agent policy, backtesting & visualization

---

## Project Overview
This project implements an intelligent trading agent that decides Buy / Sell / Hold using historical stock data in an offline simulated trading environment**.

### AI Techniques Used
1) **Statistical Learning (Supervised ML):**  
   A Random Forest classifier predicts next-day price movement direction (Up / Down) from technical indicators.
2) **Rule-Based Decision Making (Rational Agent Policy):**  
   A rule-based policy uses the model probability + trend rules (moving averages) + risk controls (stop-loss / take-profit) to decide Buy / Sell / Hold.

---

## PEAS Framework
  - Performance Measure (P):
      - Prediction performance: Accuracy, F1-score (Up class)  
      - Trading performance: Total Return (%), Max Drawdown (%), # Trades
  - Environment (E): Historical OHLCV stock data (offline backtest)
  - Actuators (A): Buy / Sell / Hold (max 1 share)
  - Sensors (S): OHLCV + engineered indicators (SMA, RSI, volatility, momentum, volume change)

---

## Setup & Installation (Windows PowerShell)
From the repository root folder in terminal:

'''

python -m venv .venv


.venv\Scripts\Activate.ps1


pip install -r requirements.txt

'''

If it shows error when trying to create environment: 
Allow local scripts in PowerShell

1.Open PowerShell as Administrator
2.Start menu → type “PowerShell”
3.Right-click → Run as Administrator

4.Run this command:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

When it asks, type: Y


5.Close the admin PowerShell, open a normal PowerShell in your repo, then activate:

.venv\Scripts\Activate.ps1


You should now see (.venv) at the start of your terminal line.

## Lastly, type in terminal: 

streamlit run app.py


------
#How to Run (AAPL Example)

### 1. In terminal
1) Download data (optional bc it is already there on the data folder)

Downloads daily OHLCV from Yahoo Finance and saves a CSV:

'''
python -m src.data --ticker AAPL --start 2018-01-01 --end 2025-01-01
'''

If you already have a CSV that you have yourself,you need to place place it at data/raw/AAPL.csv with columns:
Date,Open,High,Low,Close,Volume
-----


2) Train the ML model (Random Forest)

'''
python -m src.train_model --ticker AAPL
'''

This will print thr validation accuracy

------
3) Evaluate the classifier on the test split

'''
python -m src.evaluate --ticker AAPL
'''

Prints test Accuracy/F1, confusion matrix, classification report
----
4) Backtest the trading agent + baselines

'''
python -m src.backtest --ticker AAPL
'''


### 2.In STREAMLIT App
1) Run in terminal:  streamlit run app.py
2) Choose dates and continue.


Prints trading performance for:

1.Agent

2.Buy & Hold baseline

3.Random baseline


Saves plot: outputs/AAPL_portfolio_comparison.png

Where to Find Outputs

Trained model: models/AAPL_rf.joblib

Backtest plot: outputs/AAPL_portfolio_comparison.png

Printed metrics: shown in the terminal after running commands


